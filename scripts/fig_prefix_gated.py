#!/usr/bin/env python3
"""Figure 4 (fig:prefix-gated) and the gating operating points quoted in
Section 4.5. Reads exp/results/protonet_ablations/prefix_results.json (written
by scripts/prefix_gating_eval.py). Writes paper/figures/fig_prefix_gated.{pdf,png}
and the mean-vs-attention-pooling companion figure.

    python scripts/fig_prefix_gated.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from _paths import ABL, FIGS, MODELS

PREFIX_SEED = 0
THETAS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
FIG_POOLS = {"tau2:g+q+d": "Tau2-Bench", "bfclv3:g+q+d": "BFCLv3", "agentclinic:g+q+d": "AgentClinic"}
BENCH_COLOR = {"tau2:g+q+d": "#0072B2", "bfclv3:g+q+d": "#E69F00", "agentclinic:g+q+d": "#009E73"}
ANNOT = (0.6, 0.7, 0.8, 0.9)
MIN_STOP = 0.10
ANNOT_OFFSET = {"tau2:g+q+d": (0, 8), "bfclv3:g+q+d": (11, 6), "agentclinic:g+q+d": (0, 7)}
ANNOT_OVERRIDE = {("tau2:g+q+d", 0.8): (2, 9), ("bfclv3:g+q+d", 0.8): (-18, -7),
                  ("bfclv3:g+q+d", 0.6): (-16, -5), ("tau2:g+q+d", 0.6): (8, 6)}
FIGSIZE = (3.3125, 2.1)
PAPER_RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
}


def curve(res_all, pool, variant):
    """(steps saved %, accuracy, stop rate) per theta for a pool's in-domain gated frontier."""
    res = res_all[f"{pool}|{variant}|s{PREFIX_SEED}"]
    ds = pool.split(":")[0]
    cc = [f"{ds}/{m}" for m in MODELS]
    xs, ys, srs = [], [], []
    for th in THETAS:
        g = [res["cells"][c]["gated"][str(th)] for c in cc if c in res["cells"]]
        xs.append(100 * np.mean([x["steps_saved"] for x in g]))
        ys.append(float(np.nanmean([x["acc"] for x in g])))
        srs.append(float(np.mean([x["stop_rate"] for x in g])))
    keep = [i for i, sr in enumerate(srs) if sr >= MIN_STOP]
    return xs, ys, srs, keep


def main():
    plt.rcParams.update(PAPER_RC)
    P = json.load(open(ABL / "prefix_results.json"))
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Confidence-gated early stopping (in-domain, prefix-trained specialist, uniform prefix):")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for pool, label in FIG_POOLS.items():
        xs, ys, srs, keep = curve(P, pool, "uniform")
        for th in (0.8, 0.9):
            i = THETAS.index(th)
            print(f"  {label:12} theta={th}: stop rate {srs[i]:.1%}, accuracy {ys[i]:.3f}, "
                  f"steps saved {xs[i]:.1f}%")
        ax.plot([xs[i] for i in keep], [ys[i] for i in keep], marker="o", ms=3.5, lw=1.6,
                color=BENCH_COLOR[pool], label=label)
        for th in ANNOT:
            i = THETAS.index(th)
            if i in keep:
                dx, dy = ANNOT_OVERRIDE.get((pool, th), ANNOT_OFFSET[pool])
                ax.annotate(rf"$\theta$={th}", (xs[i], ys[i]), xytext=(dx, dy),
                            textcoords="offset points", fontsize=9, color=BENCH_COLOR[pool],
                            ha="center", bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                                   ec="none", alpha=0.75))
    ax.set_xlabel("Steps saved (%)"); ax.set_ylabel("Early-stopped accuracy")
    ax.legend(frameon=False, loc="best", handlelength=1.6, labelspacing=0.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.97, bottom=0.21)
    fig.savefig(FIGS / "fig_prefix_gated.pdf"); fig.savefig(FIGS / "fig_prefix_gated.png", dpi=200)
    print("saved", FIGS / "fig_prefix_gated.{pdf,png}")

    # companion: mean pooling (solid) vs attention pooling (dashed)
    fig, ax = plt.subplots(figsize=(3.3125, 2.6))
    for pool, label in FIG_POOLS.items():
        for variant, ls in (("uniform", "-"), ("uniform_attn", "--")):
            key = f"{pool}|{variant}|s{PREFIX_SEED}"
            if key not in P:
                continue
            xs, ys, _, keep = curve(P, pool, variant)
            ax.plot([xs[i] for i in keep], [ys[i] for i in keep], ls, marker="o", ms=3, lw=1.5,
                    color=BENCH_COLOR[pool])
    ax.set_xlabel("Steps saved (%)"); ax.set_ylabel("Early-stopped accuracy")
    handles = ([mlines.Line2D([], [], color=BENCH_COLOR[p], lw=1.6, label=l) for p, l in FIG_POOLS.items()]
               + [mlines.Line2D([], [], color="#555555", lw=1.5, ls=ls, label=lab)
                  for ls, lab in (("-", "mean pool"), ("--", "attn pool"))])
    ax.legend(handles=handles, frameon=False, ncol=1, fontsize=9, handlelength=1.6,
              labelspacing=0.25, handletextpad=0.5, loc="upper right")
    fig.subplots_adjust(left=0.145, right=0.985, top=0.97, bottom=0.17)
    fig.savefig(FIGS / "fig_prefix_gated_attn.pdf"); fig.savefig(FIGS / "fig_prefix_gated_attn.png", dpi=200)
    print("saved", FIGS / "fig_prefix_gated_attn.{pdf,png}")


if __name__ == "__main__":
    main()
