#!/usr/bin/env python3
"""Aggregate the 25-pool x 3-seed sweep into the paper's generalization matrices.

Reads results/protonet_ablations/precog_parts/<pool>_s<seed>.json (written by
scripts/precog_sweep.py; each holds test_curve = {epoch -> {"<bench>/<model>"
-> {task -> AUROC}}}) and produces the two 7x3 matrices of the paper
(cross-model, Table 2; cross-benchmark, appendix):

  * Best-within-pool stopping: for each (pool, seed), select the evaluation
    epoch maximizing the mean over the pool's own in-distribution cells
    (task-mean AUROC per cell, then mean over cells).
  * models matrix: rows = policy combinations trained within one benchmark
    (aggregated over the three benchmarks and 3 seeds), cols = evaluated policy.
  * benchmarks matrix: rows = benchmark combinations (all three policies),
    cols = evaluated benchmark (aggregated over policies and 3 seeds).

Writes results/protonet_ablations/generalization_heatmaps_7x3.json.

  python scripts/aggregate_generalization.py
  python scripts/aggregate_generalization.py --check   # compare, don't write
"""
import argparse
import json
from pathlib import Path

import numpy as np

# repo root = parent of scripts/ (works from any cwd, no absolute paths)
REPO = Path(__file__).resolve().parent.parent
for cand in ("exp/results/protonet_ablations", "results/protonet_ablations"):
    ABL = REPO / cand
    if (ABL / "precog_parts").exists():
        break
PARTS = ABL / "precog_parts"
OUT = ABL / "generalization_heatmaps_7x3.json"

BENCHES = ["tau2", "bfclv3", "agentclinic"]
MODELS = {"d": "deepseek", "q": "qwen", "g": "gpt5mini"}
SEEDS = (0, 1, 2)
MODEL_ROWS = ["d", "q", "g", "q+d", "g+d", "g+q", "g+q+d"]
BENCH_ROWS = ["tau2:g+q+d", "bfclv3:g+q+d", "agentclinic:g+q+d",
              "bfclv3+tau2:all", "agentclinic+tau2:all",
              "agentclinic+bfclv3:all", "agentclinic+bfclv3+tau2:all"]


def load_curve(pool, seed):
    return json.load(open(PARTS / f"{pool}_s{seed}.json"))["test_curve"]


def cell_mean(tc_ep, cell):
    v = tc_ep.get(cell, {})
    return float(np.mean(list(v.values()))) if v else np.nan


def best_epoch(tc, id_cells):
    """Epoch maximizing the pool's own ID mean over the test grid."""
    eps = sorted(tc, key=int)
    return max(eps, key=lambda e: np.nanmean([cell_mean(tc[e], c) for c in id_cells]))


def models_matrix():
    M, ID = [], []
    for combo in MODEL_ROWS:
        ms = combo.split("+")
        row = []
        for evalm in (MODELS[k] for k in ("d", "q", "g")):
            vals = []
            for b in BENCHES:
                pool = f"{b}_{'-'.join(ms)}"
                for s in SEEDS:
                    tc = load_curve(pool, s)
                    e = best_epoch(tc, [f"{b}/{MODELS[m]}" for m in ms])
                    vals.append(cell_mean(tc[e], f"{b}/{evalm}"))
            row.append(float(np.nanmean(vals)))
        M.append(row)
        ID.append([MODELS[k] in {MODELS[m] for m in ms} for k in ("d", "q", "g")])
    return {"rows": MODEL_ROWS, "cols": [MODELS[k] for k in ("d", "q", "g")],
            "matrix": M, "id": ID}


def benchmarks_matrix():
    M, ID = [], []
    for r in BENCH_ROWS:
        poolspec, mods = r.split(":")
        benches = poolspec.split("+")
        pool = (f"{benches[0]}_{'-'.join(mods.split('+'))}" if len(benches) == 1
                else f"{'-'.join(sorted(benches))}_all")
        row = []
        for eb in BENCHES:
            vals = []
            for s in SEEDS:
                tc = load_curve(pool, s)
                idc = [f"{b}/{m}" for b in benches for m in MODELS.values()]
                e = best_epoch(tc, idc)
                vals += [cell_mean(tc[e], f"{eb}/{m}") for m in MODELS.values()]
            row.append(float(np.nanmean(vals)))
        M.append(row)
        ID.append([b in benches for b in BENCHES])
    return {"rows": BENCH_ROWS, "cols": BENCHES, "matrix": M, "id": ID}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the existing JSON instead of writing")
    args = ap.parse_args()

    out = {"models": models_matrix(), "benchmarks": benchmarks_matrix()}

    if args.check:
        ref = json.load(open(OUT))
        for key in ("models", "benchmarks"):
            d = np.abs(np.array(out[key]["matrix"]) - np.array(ref[key]["matrix"])).max()
            assert d < 1e-12, f"{key}: max diff {d}"
            assert out[key]["rows"] == ref[key]["rows"]
            assert out[key]["id"] == ref[key]["id"]
            print(f"{key}: exact match (max diff {d})")
        return

    json.dump(out, open(OUT, "w"), indent=1)
    for key in ("models", "benchmarks"):
        print(f"{key}: rows={out[key]['rows']}")
        for r, row in zip(out[key]["rows"], out[key]["matrix"]):
            print(f"  {r:28s} " + "  ".join(f"{v:.3f}" for v in row))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
