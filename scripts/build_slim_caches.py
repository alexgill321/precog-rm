#!/usr/bin/env python3
"""Build the per-cell slim caches from the released embedding corpora.

This is the (previously notebook-only) preprocessing step between the raw
embedding release and every PrecogRM experiment: for each (benchmark, model)
cell it loads the step embeddings, forms the canonical per-task clustering
(HDBSCAN over cosine of the concatenated [action|observation] step vector),
keeps mixed-outcome tasks, and writes a compact float16 pickle to
results/protonet_ablations/slim/<benchmark>_<model>.pkl.

Afterwards, convert slim -> memory-mapped caches (what merged_cv_table.py and
the sweep actually read):

    python scripts/pool_traj_worker.py --convert

Expected data layout (the HuggingFace release, see data/table1_data/PLACEHOLDER.md):

    data/table1_data/<benchmark>_<model>/
        <task dirs with reward stubs>
        embeddings/action/<task>.npz
        embeddings/response/<task>.npz

Usage:
    python scripts/build_slim_caches.py                 # all 9 cells
    python scripts/build_slim_caches.py --cells tau2:qwen bfclv3:deepseek
"""
import argparse
import gc
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.prm.cluster_sequence import load_all_tasks              # noqa: E402
from src.prm.paired_action_response import hdbscan_cluster_affinity  # noqa: E402

DATA_ROOT = REPO / "data" / "table1_data"
ABL = REPO / "exp" / "results" / "protonet_ablations"
SLIM_DIR = ABL / "slim"

DATASETS = ["agentclinic", "bfclv3", "tau2"]
MODELS = ["gpt5mini", "qwen", "deepseek"]

# protocol constants — identical to the paper pipeline; do not change
SR_LO, SR_HI = 0.16, 0.84
MAX_ROLLOUTS, MIN_CLUSTER_SIZE, NOISE_STRATEGY = 50, 5, "outlier_bin"
DROP_LAST_STEP = {"agentclinic": True, "bfclv3": False, "tau2": False}


def _cluster_steps(X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    Xu = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    labels = hdbscan_cluster_affinity(Xu @ Xu.T, min_cluster_size=MIN_CLUSTER_SIZE,
                                      min_samples=None, noise_strategy=NOISE_STRATEGY)
    return labels, int(labels.max()) + 1


def _prep_task(t, drop):
    """Per-rollout features: pooled paired vector, step sequences, canonical
    concat-paired cluster path. `drop` removes the terminal step (AgentClinic
    diagnosis-leakage control)."""
    if t.step_embeddings is None or not t.rid_to_emb_rows:
        return None
    D = t.step_embeddings.shape[1]
    d = D // 2

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    kept = {}
    for r, rows in t.rid_to_emb_rows.items():
        if not rows or r not in t.rewards:
            continue
        kept[r] = rows[:-1] if (drop and len(rows) > 1) else rows
    rids = sorted(kept)
    if len(rids) < 6:
        return None
    order = [(r, i) for r in rids for i in range(len(kept[r]))]
    labels, K = _cluster_steps(t.step_embeddings[[kept[r][i] for r, i in order]])
    feats = {"label": t.label, "K": K, "pooled": {}, "aseq": {}, "rseq": {},
             "path": {}, "y": {}}
    off = 0
    for r in rids:
        rows = kept[r]
        E = t.step_embeddings[rows]
        a, s = E[:, :d], E[:, d:]
        feats["pooled"][r] = unit(E.mean(0))
        feats["aseq"][r] = a
        feats["rseq"][r] = s
        L = len(rows)
        feats["path"][r] = [int(labels[off + i]) for i in range(L)]
        off += L
        feats["y"][r] = 1 if t.rewards[r] >= 0.5 else 0
    feats["rids"] = np.array(rids)
    feats["sr"] = float(np.mean([feats["y"][r] for r in rids]))
    return feats


def _slim(cell):
    """Mixed-outcome tasks only; step sequences to float16."""
    out = []
    for t in cell:
        if not (SR_LO <= t["sr"] <= SR_HI):
            continue
        out.append({"label": t["label"], "K": t["K"], "sr": t["sr"],
                    "rids": t["rids"], "y": t["y"], "path": t["path"],
                    "pooled": t["pooled"],
                    "aseq": {r: a.astype(np.float16) for r, a in t["aseq"].items()},
                    "rseq": {r: a.astype(np.float16) for r, a in t["rseq"].items()}})
    return out


def build_cell(dataset, model):
    src = DATA_ROOT / f"{dataset}_{model}"
    if not src.exists():
        print(f"  [skip] {dataset}/{model}: {src} not found "
              f"(download the data release first)")
        return
    pk = SLIM_DIR / f"{dataset}_{model}.pkl"
    if pk.exists():
        print(f"  [skip] {dataset}/{model}: {pk.name} already built")
        return
    t0 = time.time()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tasks = load_all_tasks(str(src), max_rollouts=MAX_ROLLOUTS,
                               skip_clustering=True)
    drop = DROP_LAST_STEP[dataset]
    cell = [c for c in (_prep_task(t, drop) for t in tasks) if c is not None]
    slim = _slim(cell)
    del cell, tasks
    gc.collect()
    SLIM_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(slim, open(pk, "wb"))
    print(f"  {dataset}/{model}: {len(slim)} mixed tasks in "
          f"{time.time() - t0:.0f}s -> {pk.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*",
                    help="subset like tau2:qwen (default: all 9)")
    args = ap.parse_args()
    cells = ([tuple(c.split(":")) for c in args.cells] if args.cells
             else [(d, m) for d in DATASETS for m in MODELS])
    for d, m in cells:
        build_cell(d, m)
    print("done. Next: python scripts/pool_traj_worker.py --convert")


if __name__ == "__main__":
    main()
