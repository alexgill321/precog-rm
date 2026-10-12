#!/usr/bin/env python3
"""Appendix B, tab:channel: where does the outcome signal live -- actions,
observations, or both (disjoint vs joint channel treatment)?

For every (benchmark, policy) cell and each channel view
  action      a_t only          observation  o_t only
  independent HDBSCAN per channel, interleaved state ids over the union vocab
  paired      joint [a_t || o_t] step vector (the paper's canonical construction)
we report the RAW ceiling (best of NC, RBF on the channel's pooled vector) and
the CLUSTERED ceiling (best fastText n in {1,2,5} on state-id paths clustered
from that channel's own step embeddings; HDBSCAN cosine, min_cluster_size=5,
unassigned steps -> one outlier state). Per task: stratified 3-fold CV
(KFold(3, shuffle, seed 42)); aggregation: mean over tasks within each fold,
then mean ± std across the three folds. Evaluation sets are the Table 1 sets
(held-out test tasks, >=5 rollouts per class, AgentClinic tool-reliant).

Reads the mmap caches; writes exp/results/protonet_ablations/channel_ablation.json.
fastText training is multi-threaded and unseeded, so clustered-ceiling values
reproduce to within about ±0.01.

    python scripts/q4_channel_ablation.py                     # all 9 cells
    python scripts/q4_channel_ablation.py --cells tau2:qwen   # subset
"""
import argparse
import json
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from _paths import ABL, MMAP_DIR, SPLITS, DATASETS, MODELS, BENCH_NAME, MODEL_NAME, require

warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = ABL / "channel_ablation.json"
N_FOLDS, SEED, MCS = 3, 42, 5
CHANNELS = ["action", "observation", "independent", "paired"]


def _unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-9)


def _cluster(X):
    from src.prm.paired_action_response import hdbscan_cluster_affinity
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    Xu = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    labels = hdbscan_cluster_affinity(Xu @ Xu.T, min_cluster_size=MCS, min_samples=None,
                                      noise_strategy="outlier_bin")
    return labels, int(labels.max()) + 1


def _single_channel_paths(t, key):
    rids = [int(r) for r in t["rids"]]
    order = [(r, i) for r in rids for i in range(len(t[key][r]))]
    labels, K = _cluster(np.stack([t[key][r][i] for r, i in order]))
    paths, off = {}, 0
    for r in rids:
        L = len(t[key][r]); paths[r] = [int(labels[off + i]) for i in range(L)]; off += L
    return paths, K


def channel_view(t, chan):
    """(vector per rollout, state-id path per rollout, K) for one channel view."""
    rids = [int(r) for r in t["rids"]]
    if chan == "paired":
        vec = {r: np.asarray(t["pooled"][r], dtype=np.float64) for r in rids}
        return vec, {r: list(t["path"][r]) for r in rids}, t["K"]
    if chan == "action":
        vec = {r: _unit(np.asarray(t["aseq"][r], dtype=np.float64).mean(0)) for r in rids}
        paths, K = _single_channel_paths(t, "aseq")
        return vec, paths, K
    if chan == "observation":
        vec = {r: _unit(np.asarray(t["rseq"][r], dtype=np.float64).mean(0)) for r in rids}
        paths, K = _single_channel_paths(t, "rseq")
        return vec, paths, K
    # independent: pool all 2T single-channel steps in the shared d-space; two vocabularies
    vec = {r: _unit(np.concatenate([np.asarray(t["aseq"][r], dtype=np.float64),
                                    np.asarray(t["rseq"][r], dtype=np.float64)], 0).mean(0)) for r in rids}
    a_paths, Ka = _single_channel_paths(t, "aseq")
    o_paths, Ko = _single_channel_paths(t, "rseq")
    paths = {r: [x for a, o in zip(a_paths[r], o_paths[r]) for x in (a, o + Ka)] for r in rids}
    return vec, paths, Ka + Ko


def fold_aurocs(t, vec, paths, K):
    """{method: [auroc fold0, fold1, fold2]} (nan where a fold lacks both classes)."""
    from src.prm.cluster_sequence import FastTextPRM, _aggregate_scores
    rids = np.array([int(r) for r in t["rids"]])
    y = np.array([t["y"][r] for r in rids])
    P = np.nan_to_num(np.stack([vec[r] for r in rids]))
    out = {m: [] for m in ("NC", "RBF", "FT1", "FT2", "FT5")}
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(rids):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            for m in out: out[m].append(np.nan)
            continue
        cp, cf = P[tr][y[tr] == 1].mean(0), P[tr][y[tr] == 0].mean(0)
        cp /= max(np.linalg.norm(cp), 1e-9); cf /= max(np.linalg.norm(cf), 1e-9)
        out["NC"].append(roc_auc_score(y[te], P[te] @ cp - P[te] @ cf))
        svm = SVC(kernel="rbf", gamma="scale", C=1.0).fit(P[tr], y[tr])
        out["RBF"].append(roc_auc_score(y[te], svm.decision_function(P[te])))
        for n in (1, 2, 5):
            ft = FastTextPRM(wordNgrams=n, dim=20)
            ft.fit({int(r): paths[int(r)] for r in rids[tr]}, {int(r): int(t["y"][int(r)]) for r in rids[tr]}, K)
            sc = [_aggregate_scores(ft.score_path(paths[int(r)], rid=int(r)), "final") for r in rids[te]]
            out[f"FT{n}"].append(roc_auc_score(y[te], sc) if np.all(np.isfinite(sc)) else np.nan)
    return out


def aggregate(per_task):
    """per_task: {label: {method: [folds]}} -> {method: (mean, std across fold-level means)}"""
    res = {}
    for m in ("NC", "RBF", "FT1", "FT2", "FT5"):
        M = np.array([v[m] for v in per_task.values()], dtype=float)
        fm = np.nanmean(M, axis=0)
        res[m] = (float(np.nanmean(fm)), float(np.nanstd(fm)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-only", action="store_true", help="print the table from channel_ablation.json")
    args = ap.parse_args()
    R = json.load(open(OUT)) if OUT.exists() else {}
    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else [(d, m) for d in DATASETS for m in MODELS]
    todo = [(d, m) for d, m in cells if args.force or any(c not in R.get(f"{d}/{m}", {}) for c in CHANNELS)]
    if args.print_only:
        todo = []
    if todo:
        require(MMAP_DIR, "mmap caches", "caches")
        import pool_traj_worker as W
        W.load_all_cells()
        test = {ds: set(v) for ds, v in json.load(open(SPLITS)).items()}
    for ds, m in todo:
        ck = f"{ds}/{m}"
        tasks = W.eval_tasks_for(ds, m, test)
        for chan in CHANNELS:
            if ck in R and chan in R[ck] and not args.force:
                continue
            per = {}
            for t in tasks:
                vec, paths, K = channel_view(t, chan)
                per[t["label"]] = fold_aurocs(t, vec, paths, K)
            agg = aggregate(per)
            R.setdefault(ck, {})[chan] = {"n": len(per), "agg": agg, "per_task": per}
            raw = max(agg["NC"][0], agg["RBF"][0]); clu = max(agg[f"FT{n}"][0] for n in (1, 2, 5))
            print(f"  {ck:22} {chan:12} n={len(per):3d}  raw ceiling={raw:.3f}  clustered ceiling={clu:.3f}", flush=True)
            json.dump(R, open(OUT, "w"))

    print(f"\n{'benchmark':12} {'policy':11} {'N':>3} | raw: " + " ".join(f"{c[:5]:>6}" for c in CHANNELS)
          + " | clustered: " + " ".join(f"{c[:5]:>6}" for c in CHANNELS))
    for ds in ("tau2", "bfclv3", "agentclinic"):
        for m in ("gpt5mini", "qwen", "deepseek"):
            ck = f"{ds}/{m}"
            if ck not in R or any(c not in R[ck] for c in CHANNELS):
                continue
            raw = [max(R[ck][c]["agg"]["NC"][0], R[ck][c]["agg"]["RBF"][0]) for c in CHANNELS]
            clu = [max(R[ck][c]["agg"][f"FT{n}"][0] for n in (1, 2, 5)) for c in CHANNELS]
            print(f"{BENCH_NAME[ds]:12} {MODEL_NAME[m]:11} {R[ck]['paired']['n']:3d} |      "
                  + " ".join(f"{v:6.3f}" for v in raw) + " |            " + " ".join(f"{v:6.3f}" for v in clu))


if __name__ == "__main__":
    main()
