#!/usr/bin/env python3
"""Figure 5 (fig:training-curves): one figure*-width row with (a) mean training
reward, (b) held-out pass@1 with the base-model reference line, (c) cumulative
FLOPs consumed as % of baseline at the same step, (d) stacked legend.
Color = support type, linestyle = theta, linewidth = m_c.

Reads the official runs through grpo_common.load_all(); writes
paper/figures/fig_grpo_curves.{pdf,png}.

    python scripts/fig_grpo_curves.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from _paths import FIGS
from grpo_common import VARIANTS, MATCHED, load_all, base_model_pass_rate

FIG_W, FIG_H = 7.0, 2.15
WIDTH_RATIOS = [1, 1, 1, 0.62]
COLOR = {"baseline": "#6b7280", "rolling": "#0072B2", "permanent": "#E69F00"}
LS = {None: "-", 0.8: "-", 0.7: "--", 0.65: ":"}
LW = {None: 2.0, 4: 1.7, 0: 1.2}
PAPER_RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
}


def style(v):
    vc = VARIANTS[v]
    return dict(color=COLOR[vc["support"]], ls=LS[vc["theta"]], lw=LW[vc["mc"]])


def main():
    plt.rcParams.update(PAPER_RC)
    data = load_all()
    base_pass = base_model_pass_rate()

    def stack(v, col):
        return pd.concat([r.set_index("batch")[col] for r in data[v]], axis=1).sort_index()

    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H), gridspec_kw=dict(width_ratios=WIDTH_RATIOS))
    ax_r, ax_p, ax_f, ax_leg = axes

    for v in VARIANTS:                                   # (a) training reward
        arr = stack(v, "reward")
        m = arr.mean(axis=1).rolling(5, min_periods=1).mean()
        ax_r.plot(m.index, m.values, **style(v))
        if v == "baseline":
            sem = (arr.std(axis=1) / np.sqrt(arr.shape[1])).rolling(5, min_periods=1).mean()
            ax_r.fill_between(m.index, m - sem, m + sem, color=COLOR["baseline"], alpha=0.18, lw=0)
    ax_r.set_ylabel("Mean Reward"); ax_r.set_title("(a) Training Reward")

    for v in VARIANTS:                                   # (b) held-out pass@1
        arr = stack(v, "pass_rate").dropna(how="all")
        m = arr.mean(axis=1)
        ax_p.plot(m.index, m.values, marker="o", ms=2.4, **style(v))
        if v == "baseline":
            ax_p.fill_between(m.index, arr.min(axis=1), arr.max(axis=1), color=COLOR["baseline"],
                              alpha=0.15, lw=0)
    if base_pass is not None:
        ax_p.axhline(base_pass, color="#9ca3af", lw=1.0, ls=(0, (2, 2)))
    ax_p.set_ylabel("Pass@1 (%)"); ax_p.set_title("(b) Eval Pass@1")

    def cum_frame(v):                                    # (c) cumulative FLOPs, % of baseline
        return (pd.concat([r.set_index("batch").flops.sort_index().cumsum() for r in data[v]], axis=1)
                  .sort_index().ffill())
    base_mean = cum_frame("baseline").mean(axis=1)
    ax_f.axhline(100, **style("baseline"))
    for v in VARIANTS:
        if v == "baseline":
            continue
        consumed = 100 * cum_frame(v).reindex(base_mean.index).ffill().div(base_mean, axis=0)
        ax_f.plot(consumed.mean(axis=1).index, consumed.mean(axis=1).values, **style(v))
    ax_f.set_ylabel("% of Baseline FLOPs"); ax_f.set_title("(c) FLOPs Consumed")

    for ax in (ax_r, ax_p, ax_f):
        ax.set_xlabel("Step"); ax.set_xlim(0, MATCHED)

    ax_leg.axis("off")                                   # (d) stacked legend
    handles = [mlines.Line2D([], [], **style(v), label=VARIANTS[v]["label"]) for v in VARIANTS]
    ax_leg.legend(handles=handles, loc="center left", ncol=1, frameon=False, handlelength=2.0,
                  labelspacing=0.7, handletextpad=0.5, borderaxespad=0.0)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.87, bottom=0.20, wspace=0.42)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_grpo_curves.pdf"); fig.savefig(FIGS / "fig_grpo_curves.png", dpi=400)
    print("saved", FIGS / "fig_grpo_curves.{pdf,png}")


if __name__ == "__main__":
    main()
