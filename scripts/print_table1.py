#!/usr/bin/env python3
"""Compute Table 1 from the released per-task, per-fold raw outputs.

Both JSONs are written by scripts/merged_cv_table.py and hold, for every
(benchmark, model) cell, method, and held-out test task, the AUROC of each of
the 3 CV folds:

  merged_cv_table.json       baselines (NC, RBF, FT1/2/5) + the generalist
                             Precog-RM (all-benchmark pool)
  merged_cv_table_spec.json  the per-benchmark specialist Precog-RM
                             (the configuration reported in Table 1)

This script performs the paper's aggregation: per fold, average over the
cell's tasks; report mean +/- std across the 3 folds; Precog-RM additionally
averaged over its 3 training seeds (PN_s0/s1/s2).

    python scripts/print_table1.py
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ABL = REPO / "exp/results/protonet_ablations"

BENCH = {"tau2": "Tau2-Bench", "bfclv3": "BFCLv3", "agentclinic": "AgentClinic"}
MODEL = {"deepseek": "DS-V3.2", "qwen": "Qwen3-80B", "gpt5mini": "GPT-5-mini"}
ORDER = [f"{b}/{m}" for b in ("tau2", "bfclv3", "agentclinic")
         for m in ("deepseek", "qwen", "gpt5mini")]
COLS = ["NC", "RBF", "FT1", "FT2", "FT5", "PN"]


def fold_stats(per_task):
    """{task: [f0, f1, f2]} -> (mean, std) across fold-level task-means."""
    M = np.array(list(per_task.values()), dtype=float)   # (tasks, folds)
    fold_means = np.nanmean(M, axis=0)
    return float(np.nanmean(fold_means)), float(np.nanstd(fold_means)), M.shape[0]


def main():
    base = json.load(open(ABL / "merged_cv_table.json"))
    spec = json.load(open(ABL / "merged_cv_table_spec.json"))
    print(f"{'cell':24} " + " ".join(f"{c:>12}" for c in COLS) + f" {'N':>4}")
    for cell in ORDER:
        row, n_tasks, pn = {}, 0, []
        for meth, per_task in base.get(cell, {}).items():
            if meth.startswith("PN"):
                continue                       # generalist PN; Table 1 reports the specialist
            mu, sd, n = fold_stats(per_task)
            row[meth] = (mu, sd)
            n_tasks = max(n_tasks, n)
        for meth, per_task in spec.get(cell, {}).items():
            if meth.startswith("PN"):
                mu, sd, n = fold_stats(per_task)
                pn.append((mu, sd))
                n_tasks = max(n_tasks, n)
        if pn:
            row["PN"] = (float(np.mean([x[0] for x in pn])),
                         float(np.mean([x[1] for x in pn])))
        label = f"{BENCH[cell.split('/')[0]]}/{MODEL[cell.split('/')[1]]}"
        cells = " ".join(
            f"{row[c][0]:.3f}±{row[c][1]:.3f}" if c in row else f"{'—':>12}"
            for c in COLS)
        print(f"{label:24} {cells} {n_tasks:>4}")


if __name__ == "__main__":
    main()
