#!/usr/bin/env python3
"""Untrained-encoder reference row of the cross-benchmark table (App. J).

A randomly initialised Precog-RM encoder (the sweep architecture: no positional
embeddings, no type flag, d'=128, L=2) is evaluated under the standard support
protocol (k=4 per class, 5 seeded draws) on every cell's held-out tasks, for
initialisation seeds 0-2. Writes exp/results/protonet_ablations/untrained_precog.json
({cell: [auroc_s0, auroc_s1, auroc_s2]}) and prints the per-benchmark means
(mean over the benchmark's three policy cells and the three seeds).

    python scripts/eval_untrained.py            # compute (or reuse cache) and print
    python scripts/eval_untrained.py --force    # recompute
    python scripts/eval_untrained.py --check    # recompute and compare with the shipped JSON
"""
import argparse
import json

import numpy as np

from _paths import ABL, MMAP_DIR, SPLITS, DATASETS, MODELS, BENCH_NAME, require

OUT = ABL / "untrained_precog.json"
SEEDS = (0, 1, 2)


def compute(device="cpu"):
    import torch
    import pool_traj_worker as W
    from src.prm.train_precog import PrecogProtoNet
    require(MMAP_DIR, "mmap caches", "caches")
    W.load_all_cells()
    test = {ds: set(v) for ds, v in json.load(open(SPLITS)).items()}
    out = {f"{ds}/{m}": [] for ds in DATASETS for m in MODELS}
    if device == "cpu":
        PrecogProtoNet._select_device = staticmethod(lambda: torch.device("cpu"))
    # the network is built inside fit_from_embeddings; epochs=0 builds it and skips training.
    # Any pool with both classes works (initialisation only depends on the seed and d=3072).
    t0 = W.train_tasks_for("tau2", "qwen", test)[0]
    rids = [int(r) for r in t0["rids"]]
    act = [t0["aseq"][r] for r in rids]; res = [t0["rseq"][r] for r in rids]
    lab = [int(t0["y"][r]) for r in rids]; tid = [t0["label"]] * len(rids)
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        prm = PrecogProtoNet(**{**W.PROTO_KWARGS, "epochs": 0})
        prm.fit_from_embeddings(act, res, lab, tid)
        for ds in DATASETS:
            for m in MODELS:
                per = [W.eval_protonet_task(prm, t) for t in W.eval_tasks_for(ds, m, test)]
                per = [a for a in per if not np.isnan(a)]
                out[f"{ds}/{m}"].append(float(np.mean(per)))
                print(f"  seed {seed} {ds}/{m}: {len(per)} tasks AUROC={np.mean(per):.3f}", flush=True)
    return out


def print_row(u):
    print("Untrained encoder, held-out AUROC by evaluated benchmark (mean over policy cells and 3 seeds):")
    for b in ("tau2", "bfclv3", "agentclinic"):
        print(f"  {BENCH_NAME[b]:12} {np.mean([np.mean(u[f'{b}/{m}']) for m in MODELS]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu gives bit-reproducible random init")
    args = ap.parse_args()
    if OUT.exists() and not (args.force or args.check):
        print_row(json.load(open(OUT)))
        return
    u = compute(args.device)
    if args.check and OUT.exists():
        old = json.load(open(OUT))
        diff = max(abs(np.mean(u[c]) - np.mean(old[c])) for c in u)
        print(f"check: max |cell-mean difference| vs shipped JSON = {diff:.4f}")
        print_row(u)
        return
    json.dump(u, open(OUT, "w"), indent=1)
    print_row(u)


if __name__ == "__main__":
    main()
