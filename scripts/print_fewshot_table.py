#!/usr/bin/env python3
"""Appendix H (tab:fewshot-full) and the numbers behind Figure 3.

Aggregates exp/results/protonet_ablations/support_sweep.json (written by
scripts/support_sweep.py): mean per-task AUROC per cell, then mean over the
benchmark's three policy cells; Precog-RM averaged over its 3 training seeds.

    python scripts/print_fewshot_table.py           # table
    python scripts/print_fewshot_table.py --json    # machine-readable (used by fig_fewshot_bars.py)
"""
import argparse
import json

import numpy as np

from _paths import ABL, BENCH_NAME, MODELS

KS = ["1", "2", "4", "8"]
METHODS = ["ProtoNet", "NC", "RBF", "FT1"]


def aggregate(path=ABL / "support_sweep.json"):
    s = json.load(open(path))
    out = {}
    for b in ("tau2", "bfclv3", "agentclinic"):
        out[b] = {}
        for k in KS:
            row = {}
            for meth in METHODS:
                vals = []
                for m in MODELS:
                    d = s[f"{b}/{m}"][k]
                    if meth == "ProtoNet":
                        vals.append(np.mean([np.mean(list(d[f"ProtoNet_s{i}"].values()))
                                             for i in (0, 1, 2)]))
                    else:
                        vals.append(np.mean(list(d[meth].values())))
                row[meth] = float(np.mean(vals))
            out[b][k] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    agg = aggregate()
    if args.json:
        print(json.dumps(agg, indent=1))
        return
    print(f"{'benchmark':12} {'k':>2} " + " ".join(f"{m:>8}" for m in METHODS) + f" {'best-bl':>8}")
    for b, rows in agg.items():
        for k, row in rows.items():
            best = max(row[m] for m in METHODS[1:])
            print(f"{BENCH_NAME[b]:12} {k:>2} " + " ".join(f"{row[m]:8.3f}" for m in METHODS)
                  + f" {best:8.3f}")


if __name__ == "__main__":
    main()
