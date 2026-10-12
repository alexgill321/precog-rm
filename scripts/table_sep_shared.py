#!/usr/bin/env python3
"""Appendix L (tab:sep-shared): separability on the held-out tasks shared by
all three policies of a benchmark.

Reads the per-task, per-fold AUROCs written by scripts/merged_cv_table.py
(exp/results/protonet_ablations/merged_cv_table{,_spec}.json), intersects the
task sets of the three policy cells of each benchmark, and reports the mean over
the shared tasks within each fold, then over the 3 folds (the Table 1 convention) (Precog-RM: per-benchmark specialist,
averaged over its 3 training seeds).

    python scripts/table_sep_shared.py
"""
import json

import numpy as np

from _paths import ABL, BENCH_NAME, MODEL_NAME

COLS = ["NC", "RBF", "FT1", "FT2", "FT5"]


def main():
    base = json.load(open(ABL / "merged_cv_table.json"))
    spec = json.load(open(ABL / "merged_cv_table_spec.json"))
    print(f"{'benchmark':12} {'policy':11} {'N':>3} " + " ".join(f"{c:>6}" for c in COLS) + f" {'PN':>6}")
    for b in ("tau2", "bfclv3", "agentclinic"):
        cells = [f"{b}/{m}" for m in ("deepseek", "qwen", "gpt5mini")]
        shared = sorted(set.intersection(*[set(base[c]["NC"]) for c in cells]))
        for c in cells:
            # Table 1 convention: mean over tasks within each fold, then mean over the 3 folds
            agg = lambda d: float(np.nanmean(np.nanmean(np.array([d[t] for t in shared], dtype=float), axis=0)))
            vals = [agg(base[c][meth]) for meth in COLS]
            pn = np.mean([agg(spec[c][f"PN_s{s}"]) for s in (0, 1, 2)])
            print(f"{BENCH_NAME[b]:12} {MODEL_NAME[c.split('/')[1]]:11} {len(shared):3d} "
                  + " ".join(f"{v:6.3f}" for v in vals) + f" {pn:6.3f}")


if __name__ == "__main__":
    main()
