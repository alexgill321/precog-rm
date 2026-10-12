#!/usr/bin/env python3
"""Appendix M (simulated early stopping): aggregate the offline replay sweep.

Reads $PRECOG_DATA/replay/results_<config>_s<seed>.json, written by
scripts/run_grpo_early_stopping_official.py (the prefix-trained controller
replayed over the completed rollouts of the three baseline GRPO seeds, theta
swept 0.55-0.95), averages the three seeds, prints the per-theta frontier for
the two strategies the paper reports (rolling, permanent) and the quoted
operating points, and writes notebooks/data/tau2_es/replay_threshold_sweep.csv
for scripts/fig_sim_pareto.py.

    python scripts/replay_sweep_table.py
"""
import json

import numpy as np
import pandas as pd

from _paths import REPLAY, GRPO_CACHE, REPO

# config name on disk -> label used by fig_sim_pareto.py / the paper
CONFIGS = {
    "grpo_support_rolling":              "GRPO rolling",
    "grpo_support_rolling_mc4":          "GRPO rolling, mc=4 (deployed)",
    "best_available_rolling":            "best-avail rolling",
    "best_available_rolling_stratified": "best-avail rolling strat.",
    "combined_rolling":                  "combined rolling",
    "combined_rolling_stratified":       "combined rolling strat.",
}
PAPER = {"GRPO rolling": "Rolling", "best-avail rolling strat.": "Permanent"}


SHIPPED = REPO / "exp/results/grpo/replay"   # copies of the 18 result files (fast path)


def load(cfg):
    base = REPLAY if (REPLAY / f"results_{cfg}_s1.json").exists() else SHIPPED
    per = [json.load(open(base / f"results_{cfg}_s{s}.json")) for s in (1, 2, 3)
           if (base / f"results_{cfg}_s{s}.json").exists()]
    if not per:
        return None
    thr = sorted(per[0].keys(), key=float)
    def arr(k):
        return np.array([[d[t]["summary"][k] for t in thr] for d in per]).mean(0)
    return dict(thr=[float(t) for t in thr], acc=arr("overall_accuracy"),
                early_acc=arr("early_accuracy"), saved=arr("compute_saved"),
                cov=arr("coverage"), bias=arr("reward_bias"), mae=arr("reward_mae"),
                n_seeds=len(per))


def main():
    rows, summ = [], {}
    for cfg, label in CONFIGS.items():
        s = load(cfg)
        if s is None:
            continue
        summ[label] = s
        for i, t in enumerate(s["thr"]):
            rows.append({"strategy": label, "θ": t, "coverage %": 100 * s["cov"][i],
                         "compute saved %": 100 * s["saved"][i],
                         "stopped acc %": 100 * s["early_acc"][i],
                         "overall acc %": 100 * s["acc"][i],
                         "reward bias": s["bias"][i], "reward MAE": s["mae"][i]})
    df = pd.DataFrame(rows)
    GRPO_CACHE.mkdir(parents=True, exist_ok=True)
    out = GRPO_CACHE / "replay_threshold_sweep.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    for label, name in PAPER.items():
        s = summ[label]
        print(f"\n== {name} support ({label}; mean of {s['n_seeds']} baseline seeds)")
        print(f"  {'θ':>5} {'acc':>6} {'stopped-acc':>11} {'saved':>6} {'coverage':>8} {'bias':>7} {'MAE':>6}")
        for i, t in enumerate(s["thr"]):
            print(f"  {t:5.2f} {s['acc'][i]:6.3f} {s['early_acc'][i]:11.3f} {s['saved'][i]:6.1%} "
                  f"{s['cov'][i]:8.1%} {s['bias'][i]:+7.3f} {s['mae'][i]:6.3f}")

    r, p = summ["GRPO rolling"], summ["best-avail rolling strat."]
    def saved_at_acc(s, target):
        # accuracy rises with theta while savings fall; interpolate savings at a target accuracy
        order = np.argsort(s["acc"])
        return float(np.interp(target, np.asarray(s["acc"])[order], np.asarray(s["saved"])[order]))
    print("\nQuoted operating points:")
    print(f"  compute saved at 84% replayed accuracy: permanent {saved_at_acc(p, .84):.1%} "
          f"vs rolling {saved_at_acc(r, .84):.1%}")
    i95 = r["thr"].index(0.95)
    print(f"  most conservative (theta=0.95): rolling saves {r['saved'][i95]:.1%} "
          f"(acc {r['acc'][i95]:.3f}), permanent {p['saved'][i95]:.1%} (acc {p['acc'][i95]:.3f})")
    for th in (0.8, 0.9):
        i = r["thr"].index(th)
        print(f"  theta={th}: reward bias rolling {r['bias'][i]:+.3f} / permanent {p['bias'][i]:+.3f}; "
              f"MAE {r['mae'][i]:.3f} / {p['mae'][i]:.3f}")


if __name__ == "__main__":
    main()
