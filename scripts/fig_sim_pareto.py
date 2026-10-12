#!/usr/bin/env python3
"""Figure 6 (fig:strictness): simulated early-stopping frontier.

Reads notebooks/data/tau2_es/replay_threshold_sweep.csv (written by
scripts/replay_sweep_table.py) and writes paper/figures/fig_sim_pareto.{pdf,png}.
Two support strategies: Rolling (support accrued from the current run) and
Permanent (pre-collected per task, "best-avail stratified").

    python scripts/fig_sim_pareto.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import GRPO_CACHE, FIGS

CSV = GRPO_CACHE / "replay_threshold_sweep.csv"
SERIES = [("GRPO rolling", "Rolling", "#0072B2", "o"),
          ("best-avail rolling strat.", "Permanent", "#E69F00", "D")]
ANNOT = (0.95, 0.85, 0.75, 0.65, 0.55)
OFF_ROLL = ((7, -3), "left", "center")
OFF_PERM = ((8, 3), "left", "bottom")
OVERRIDE = {("Permanent", 0.55): ((-4, -9), "right", "top"),
            ("Rolling", 0.95): ((8, 2), "left", "bottom"),
            ("Permanent", 0.95): None}
FIGSIZE = (3.3125, 2.5)
PAPER_RC = {
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
}


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run scripts/replay_sweep_table.py first")
    plt.rcParams.update(PAPER_RC)
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, label, color, marker in SERIES:
        d = df[df.strategy == key].sort_values("θ")
        ax.plot(d["compute saved %"], d["overall acc %"], marker=marker, ms=3.6, lw=1.6,
                color=color, label=label, zorder=3)
        for _, r in d.iterrows():
            th = round(r["θ"], 2)
            if th not in ANNOT or OVERRIDE.get((label, th), False) is None:
                continue
            (dx, dy), ha, va = OVERRIDE.get((label, th), OFF_ROLL if label == "Rolling" else OFF_PERM)
            ax.annotate(rf"$\theta$={th:g}", (r["compute saved %"], r["overall acc %"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=10,
                        color=color, ha=ha, va=va, zorder=4)
    ax.set_xlabel("Compute saved (%)"); ax.set_ylabel("Replayed accuracy (%)")
    ax.margins(x=0.13, y=0.07)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6, labelspacing=0.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.155, right=0.98, top=0.97, bottom=0.175)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_sim_pareto.pdf"); fig.savefig(FIGS / "fig_sim_pareto.png", dpi=300)
    print("saved", FIGS / "fig_sim_pareto.{pdf,png}")


if __name__ == "__main__":
    main()
