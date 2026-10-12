#!/usr/bin/env python3
"""Figure 3 (fig:fewshot): Precog-RM vs the strongest per-task baseline at
each support size k, per benchmark. Numbers come from support_sweep.json via
print_fewshot_table.aggregate(). Writes paper/figures/fig_fewshot_bars.{pdf,png}.

    python scripts/fig_fewshot_bars.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import FIGS, BENCH_NAME
from print_fewshot_table import aggregate, KS, METHODS

PN_COLOR, BL_COLOR = "#0072B2", "#b0b0b0"
YLIM = (0.55, 1.0)
FIGSIZE = (3.3125, 3.7)   # AAAI \columnwidth, included at width=\columnwidth
PAPER_RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
}


def main():
    plt.rcParams.update(PAPER_RC)
    agg = aggregate()
    data = {BENCH_NAME[b]: {"pn": [rows[k]["ProtoNet"] for k in KS],
                            "bl": [max(rows[k][m] for m in METHODS[1:]) for k in KS]}
            for b, rows in agg.items()}
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    x = np.arange(len(KS)); w = 0.38
    for ax, (name, d) in zip(axes, data.items()):
        ax.bar(x - w / 2, d["pn"], w, color=PN_COLOR, label="Precog-RM")
        ax.bar(x + w / 2, d["bl"], w, color=BL_COLOR, label="Best per-task baseline")
        ax.axhline(d["bl"][-1], color="#444444", lw=0.8, ls="--", zorder=0)
        ax.text(0.02, 0.94, name, transform=ax.transAxes, fontsize=9, fontstyle="italic", va="top")
        ax.set_ylim(*YLIM); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0)
    axes[1].set_ylabel("AUROC", labelpad=5)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(KS)
    axes[-1].set_xlabel("Support size $k$ (rollouts per class)")
    axes[2].legend(frameon=False, loc="upper right", borderaxespad=0.1,
                   handlelength=1.2, labelspacing=0.25, handletextpad=0.4)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.975, bottom=0.115, hspace=0.14)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_fewshot_bars.pdf")
    fig.savefig(FIGS / "fig_fewshot_bars.png", dpi=200)
    print("saved", FIGS / "fig_fewshot_bars.{pdf,png}")
    for name, d in data.items():
        print(f"  {name:12} PN={['%.3f' % v for v in d['pn']]} best-baseline={['%.3f' % v for v in d['bl']]}")


if __name__ == "__main__":
    main()
