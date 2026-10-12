#!/usr/bin/env python3
"""Prefix prediction + confidence gating (Figure 4 / Section 4.5 driver).

Standalone extraction of the prefix experiment from protonet_ablations.ipynb
(cells 39-40), verbatim protocol, built on scripts/pool_traj_worker.py's shared
machinery (loaders, pools, eval tasks, seeded support draws).

For a (pool, variant): load or meta-train the prefix-trained Precog-RM, then
for every evaluation cell score EVERY prefix of every held-out rollout against
support prototypes, producing (a) AUROC as a function of the observed fraction,
(b) selective commit/abstain counts per confidence threshold, and (c) the
confidence-gated early-stop frontier (accuracy of stopped rollouts vs. steps
saved, theta swept 0.55-0.95). Results merge into
exp/results/protonet_ablations/prefix_results.json, which
notebooks/fig_prefix_gated.ipynb plots.

  python scripts/prefix_gating_eval.py                          # paper config
  python scripts/prefix_gating_eval.py --variants uniform uniform_attn
  python scripts/prefix_gating_eval.py --cells agentclinic:gpt5mini --force
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.pool_traj_worker as W                      # noqa: E402
from src.prm.train_precog import PrecogProtoNet           # noqa: E402

from scripts._paths import ABL, CKPT_PRECOG, CKPT_PREFIX  # noqa: E402
CKPT_DIR = CKPT_PREFIX
PREFIX_RES_PATH = ABL / "prefix_results.json"

# ---- experiment config (identical to the notebook) ----
PREFIX_VARIANTS = {
    "full":    dict(),                      # complete queries (reuses precog sweep ckpts)
    "uniform": dict(prefix_training=True,   # query prefix l ~ U{1..T}
                    kw=dict(k_support=8, q_query=8)),
    "uniform_pos50k": dict(prefix_training=True,
                           kw=dict(no_pos=False, epochs=50, episodes_per_epoch=1000)),
    "uniform_attn":   dict(prefix_training=True, pooling="attention",
                           kw=dict(k_support=8, q_query=8)),
    "stepmatch":      dict(stepmatch_training=True, kw=dict(k_support=8, q_query=8)),
    "stepmatch_attn": dict(stepmatch_training=True, pooling="attention",
                           kw=dict(k_support=8, q_query=8)),
}
PREFIX_EVAL_PROTO = {"stepmatch": "step", "stepmatch_attn": "step"}   # default "full"
PREFIX_POOLS = ["tau2:g+q+d", "bfclv3:g+q+d", "agentclinic:g+q+d"]
PREFIX_SEED = 0
PREFIX_KW = dict(hidden_dim=128, n_heads=4, n_layers=2, ffn_dim=512, dropout=0.3,
                 epochs=100, episodes_per_epoch=100, lr=1e-4,
                 k_support=8, q_query=8, lambda_adv=0.0)
FRACTIONS = [round(0.05 * i, 2) for i in range(1, 21)]
THETAS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
KSUP_EVAL = 8


def _pfx_name(pool, variant, seed=PREFIX_SEED):
    return f"prefix_{pool.replace(':', '_').replace('+', '-')}_{variant}_s{seed}"


def train_or_load_prefix(pool, variant, test_labels, seed=PREFIX_SEED,
                         allow_train=False):
    ckpt = CKPT_DIR / f"{_pfx_name(pool, variant, seed)}.pt"
    if ckpt.exists():
        return PrecogProtoNet.load(str(ckpt))
    if variant == "full":   # identical config to the precog sweep -> reuse its ckpt
        sweep = CKPT_PRECOG / f"precog_{pool.replace(':', '_').replace('+', '-')}_s{seed}.pt"
        if sweep.exists():
            print(f"  [reuse] {sweep.name}")
            return PrecogProtoNet.load(str(sweep))
    if not allow_train:
        print(f"  [skip] no checkpoint: {_pfx_name(pool, variant, seed)} "
              f"(pass --train to meta-train it)")
        return None
    act, res, lab, tid, *_ = W._gather(W.POOLS[pool], test_labels, max_per_task=None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = dict(PREFIX_VARIANTS[variant])
    kw = {**PREFIX_KW, **spec.pop("kw", {})}
    prm = PrecogProtoNet(**kw)
    for a, v in spec.items():
        setattr(prm, a, v)
    t0 = time.time()
    prm.fit_from_embeddings(act, res, lab, tid)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    prm.save(str(ckpt))
    print(f"  trained {_pfx_name(pool, variant, seed)} in {(time.time()-t0)/60:.1f} min")
    return prm


def _encode_prefixes(prm, t, batch=64):
    """For every rollout r: embeddings of ALL prefixes 1..T_r -> {r: (T_r, H)}."""
    jobs, owners = [], []
    for r in t["rids"]:
        r = int(r)
        a, s = t["aseq"][r], t["rseq"][r]
        for l in range(1, len(a) + 1):
            jobs.append(prm._interleave_rollout(a[:l], s[:l]))
            owners.append(r)
    out = {}
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            E = prm._encode_batch([j[0] for j in jobs[i:i+batch]],
                                  [j[1] for j in jobs[i:i+batch]]).cpu().numpy()
            for e, r in zip(E, owners[i:i+batch]):
                out.setdefault(r, []).append(e)
    return {r: np.stack(v) for r, v in out.items()}


def _proto_paths(P, rids, idx):
    """(T_max, H) prototype path: mean over support rollouts of their length-t
    prefix embedding, clamped at each rollout's final embedding for t > T_r."""
    T_max = max(len(P[rids[i]]) for i in idx)
    padded = [np.vstack([P[rids[i]],
                         np.repeat(P[rids[i]][-1:], T_max - len(P[rids[i]]), axis=0)])
              for i in idx]
    return np.mean(padded, axis=0)


def prefix_eval_task(prm, t, k=KSUP_EVAL, fractions=FRACTIONS, thetas=THETAS,
                     proto="full"):
    """(auroc_by_fraction, selective counts, gated_by_theta) for one task."""
    P = _encode_prefixes(prm, t)
    rids = [int(r) for r in t["rids"]]
    y = np.array([t["y"][r] for r in rids])
    E_full = np.stack([P[r][-1] for r in rids])

    def probs(cp, cf, E):
        dp = np.linalg.norm(E - cp, axis=-1)
        df = np.linalg.norm(E - cf, axis=-1)
        z = np.clip(df - dp, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    frac_auc = {f: [] for f in fractions}
    sel = {th: {f: [0, 0, 0] for f in fractions} for th in thetas}
    gated = {th: {"correct": [], "stopped": [], "frac_obs": [], "steps_obs": [],
                  "steps_tot": [], "y": [], "score": []}
             for th in thetas}
    for sup, qry in W.support_draws(t, k):
        if proto == "step":
            cp_path = _proto_paths(P, rids, [i for i in sup if y[i] == 1])
            cf_path = _proto_paths(P, rids, [i for i in sup if y[i] == 0])
            Tp, Tf = len(cp_path), len(cf_path)

            def proto_at(l):
                return cp_path[min(l, Tp) - 1], cf_path[min(l, Tf) - 1]
        else:
            cp = E_full[sup][y[sup] == 1].mean(0)
            cf = E_full[sup][y[sup] == 0].mean(0)
        yq = y[qry]
        # (1) fraction curves + per-fraction selective counts
        for f in fractions:
            ls = [max(1, int(np.floor(f * len(P[rids[i]])))) for i in qry]
            Eq = np.stack([P[rids[i]][l - 1] for i, l in zip(qry, ls)])
            if proto == "step":
                sc = np.array([probs(*proto_at(l), Eq[j:j+1])[0]
                               for j, l in enumerate(ls)])
            else:
                sc = probs(cp, cf, Eq)
            if len(set(yq)) == 2:
                frac_auc[f].append(roc_auc_score(yq, sc))
            for th in thetas:
                mask = (sc > th) | (sc < 1 - th)
                s = sel[th][f]
                s[0] += int((((sc > 0.5).astype(int) == yq) & mask).sum())
                s[1] += int(mask.sum())
                s[2] += len(yq)
        # (2) confidence gating on stepwise probability curves
        for i in qry:
            r = rids[i]
            if proto == "step":
                pr = np.array([probs(*proto_at(l + 1), P[r][l:l+1])[0]
                               for l in range(len(P[r]))])
            else:
                pr = probs(cp, cf, P[r])
            T = len(pr)
            for th in thetas:
                hit = np.where((pr > th) | (pr < 1 - th))[0]
                crossed = bool(len(hit))
                if crossed:
                    stop = int(hit[0])
                    pred = int(pr[stop] > 0.5)
                else:
                    stop = T - 1
                    pred = int(pr[-1] > 0.5)
                g = gated[th]
                if crossed:                      # accuracy tracked on stopped ONLY
                    g["correct"].append(int(pred == t["y"][r]))
                g["stopped"].append(int(crossed))
                g["frac_obs"].append((stop + 1) / T)
                g["steps_obs"].append(stop + 1)
                g["steps_tot"].append(T)
                g["y"].append(t["y"][r])
                g["score"].append(float(pr[stop]))
    return ({f: (float(np.mean(v)) if v else np.nan) for f, v in frac_auc.items()},
            sel,
            {th: {"acc": (float(np.mean(g["correct"])) if g["correct"] else np.nan),
                  "stop_rate": float(np.mean(g["stopped"])),
                  "frac_obs": float(np.mean(g["frac_obs"])),
                  "steps_saved": 1.0 - float(np.sum(g["steps_obs"]) / np.sum(g["steps_tot"])),
                  "auroc": (float(roc_auc_score(g["y"], g["score"]))
                            if len(set(g["y"])) == 2 else np.nan)}
             for th, g in gated.items()})


def run_prefix_eval(pool, variant, test_labels, res, cells=None, seed=PREFIX_SEED,
                    force=False, allow_train=False):
    key = f"{pool}|{variant}|s{seed}"
    if key in res and not force:
        print(f"  [cached] {key}")
        return res[key]
    prm = train_or_load_prefix(pool, variant, test_labels, seed, allow_train)
    if prm is None:
        return None
    out = {"cells": {}}
    pm = PREFIX_EVAL_PROTO.get(variant, "full")
    for ds, m in (cells or W.cells_grid()):
        fa, ga = [], []
        sel_sum = {th: {f: [0, 0, 0] for f in FRACTIONS} for th in THETAS}
        for t in W.eval_tasks_for(ds, m, test_labels, k=KSUP_EVAL):
            f_auc, sel, g = prefix_eval_task(prm, t, proto=pm)
            fa.append(f_auc)
            ga.append(g)
            for th in THETAS:
                for f in FRACTIONS:
                    for j in range(3):
                        sel_sum[th][f][j] += sel[th][f][j]
        if not fa:
            continue
        out["cells"][W.cell_key(ds, m)] = {
            "n_tasks": len(fa),
            "auroc_by_fraction": {str(f): float(np.nanmean([x[f] for x in fa]))
                                  for f in FRACTIONS},
            "selective_counts": {str(th): {str(f): sel_sum[th][f] for f in FRACTIONS}
                                 for th in THETAS},
            "gated": {str(th): {kk: float(np.nanmean([g[th][kk] for g in ga]))
                                for kk in ("acc", "stop_rate", "frac_obs",
                                           "steps_saved", "auroc")}
                      for th in THETAS},
        }
        gm = out["cells"][W.cell_key(ds, m)]
        print(f"    {ds}/{m}: {gm['n_tasks']} tasks  "
              f"AUROC@0.5={gm['auroc_by_fraction']['0.5']:.3f}  "
              f"@1.0={gm['auroc_by_fraction']['1.0']:.3f}", flush=True)
    res[key] = out
    json.dump(res, open(PREFIX_RES_PATH, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=PREFIX_POOLS)
    ap.add_argument("--variants", nargs="*", default=["uniform", "uniform_attn"])
    ap.add_argument("--cells", nargs="*",
                    help="restrict eval cells, e.g. tau2:qwen (default: all 9)")
    ap.add_argument("--force", action="store_true", help="recompute cached keys")
    ap.add_argument("--train", action="store_true",
                    help="meta-train missing prefix checkpoints (expensive)")
    args = ap.parse_args()

    W.load_all_cells()
    test_labels = {ds: set(v) for ds, v in json.load(open(W.SPLITS_PATH)).items()}
    res = json.load(open(PREFIX_RES_PATH)) if PREFIX_RES_PATH.exists() else {}
    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else None
    for pool in args.pools:
        for variant in args.variants:
            print(f"{pool} | {variant}")
            run_prefix_eval(pool, variant, test_labels, res, cells=cells,
                            force=args.force, allow_train=args.train)


if __name__ == "__main__":
    main()
