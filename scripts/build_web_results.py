#!/usr/bin/env python3
"""Collect the shipped results the project page's charts read into docs/results.json.

Same aggregation as the paper scripts (print_table1.py, print_fewshot_table.py,
aggregate_generalization.py, fig_prefix_gated.py, table_grpo.py / fig_grpo_curves.py),
reduced to what the page draws: Table 1, Figure 3, Table 2 (+ the cross-benchmark
matrix), Figure 4 frontiers, Table 3 and the Figure 5 curves. Runs from the repo
alone (no data download).

    python scripts/build_web_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, ABL  # noqa: E402

OUT = REPO / "docs" / "results.json"
R = lambda x, n=3: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def table1():
    from print_table1 import fold_stats, ORDER, COLS
    base = json.load(open(ABL / "merged_cv_table.json")); spec = json.load(open(ABL / "merged_cv_table_spec.json"))
    rows = []
    for cell in ORDER:
        row, n_tasks, pn = {}, 0, []
        for meth, per_task in base.get(cell, {}).items():
            if meth.startswith("PN"): continue
            mu, sd, n = fold_stats(per_task); row[meth] = [R(mu), R(sd)]; n_tasks = max(n_tasks, n)
        for meth, per_task in spec.get(cell, {}).items():
            if meth.startswith("PN"):
                mu, sd, n = fold_stats(per_task); pn.append((mu, sd)); n_tasks = max(n_tasks, n)
        if pn: row["PN"] = [R(np.mean([x[0] for x in pn])), R(np.mean([x[1] for x in pn]))]
        best_base = max((c for c in COLS if c != "PN" and c in row), key=lambda c: row[c][0])
        b, m = cell.split("/")
        rows.append({"benchmark": b, "model": m, "n_tasks": n_tasks, "methods": row, "best_baseline": best_base})
    return {"columns": COLS, "rows": rows}


def fewshot():
    from print_fewshot_table import aggregate
    return aggregate()


def generalization():
    return json.load(open(ABL / "generalization_heatmaps_7x3.json"))


def gating():
    from fig_prefix_gated import curve, THETAS, FIG_POOLS
    P = json.load(open(ABL / "prefix_results.json"))
    out = {}
    for pool, label in FIG_POOLS.items():
        xs, ys, srs, keep = curve(P, pool, "uniform")
        out[pool.split(":")[0]] = {"label": label, "thetas": THETAS, "steps_saved": [R(x, 1) for x in xs],
                                   "accuracy": [R(y) for y in ys], "stop_rate": [R(s) for s in srs], "shown": keep}
    return out


def grpo():
    from grpo_common import VARIANTS, MATCHED, load_all, base_model_pass_rate
    import pandas as pd
    data = load_all(verbose=False)
    base_pass = base_model_pass_rate()
    tot = {v: dict(flops=np.array([r.flops.sum() for r in runs]),
                   last_eval=np.array([r.dropna(subset=["pass_rate"]).iloc[-1].pass_rate for r in runs]),
                   late_reward=np.array([r[(r.batch >= MATCHED - 10)].reward.mean() for r in runs]))
           for v, runs in data.items() if runs}
    base = tot["baseline"]["flops"].mean(); base_pass_grpo = tot["baseline"]["last_eval"].mean()
    table = []
    for v, vc in VARIANTS.items():
        if v not in tot: continue
        t = tot[v]; saved = 100 * (1 - t["flops"] / base)
        gain = ((t["last_eval"].mean() - base_pass) / (base_pass_grpo - base_pass)) if v != "baseline" else None
        table.append({"id": v, "support": vc["support"], "theta": vc["theta"], "mc": vc["mc"],
                      "flops_saved": None if v == "baseline" else [R(saved.mean(), 1), R(saved.std(), 1)],
                      "pass1": [R(t["last_eval"].mean(), 1), R(t["last_eval"].std(), 1)],
                      "retained_gain": R(gain, 2), "late_reward": [R(t["late_reward"].mean()), R(t["late_reward"].std())]})
    def stack(v, col):
        return pd.concat([r.set_index("batch")[col] for r in data[v]], axis=1).sort_index()
    cum_base = pd.concat([r.set_index("batch").flops.sort_index().cumsum() for r in data["baseline"]], axis=1).mean(axis=1)
    curves = {}
    for v in VARIANTS:
        if v not in data or not data[v]: continue
        rw = stack(v, "reward"); pr = stack(v, "pass_rate").dropna(how="all")
        cum = pd.concat([r.set_index("batch").flops.sort_index().cumsum() for r in data[v]], axis=1).reindex(cum_base.index).ffill()
        pct = 100 * cum.div(cum_base, axis=0)
        curves[v] = {"reward": {"step": [int(i) for i in rw.index], "mean": [R(x) for x in rw.mean(axis=1)], "lo": [R(x) for x in rw.min(axis=1)], "hi": [R(x) for x in rw.max(axis=1)]},
                     "pass1": {"step": [int(i) for i in pr.index], "mean": [R(x, 1) for x in pr.mean(axis=1)], "lo": [R(x, 1) for x in pr.min(axis=1)], "hi": [R(x, 1) for x in pr.max(axis=1)]},
                     "flops_pct": {"step": [int(i) for i in pct.index], "mean": [R(x, 1) for x in pct.mean(axis=1)]}}
    return {"base_pass1": R(base_pass, 1), "matched_steps": MATCHED, "table": table, "curves": curves}


def main():
    out = {"table1": table1(), "fewshot": fewshot(), "generalization": generalization(), "gating": gating(), "grpo": grpo()}
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")
    for r in out["table1"]["rows"]: print(f"  T1 {r['benchmark']}/{r['model']}: best {r['best_baseline']} {r['methods'][r['best_baseline']][0]} vs PN {r['methods'].get('PN', ['?'])[0]} (N={r['n_tasks']})")
    for r in out["grpo"]["table"]: print(f"  T3 {r['support']} θ={r['theta']} mc={r['mc']}: saved {r['flops_saved']} pass@1 {r['pass1']} retained {r['retained_gain']}")
    for b, g in out["gating"].items(): print(f"  F4 {b}: θ=0.8 saved {g['steps_saved'][g['thetas'].index(0.8)]}% acc {g['accuracy'][g['thetas'].index(0.8)]}")


if __name__ == "__main__":
    main()
