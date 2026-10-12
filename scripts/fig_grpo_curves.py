#!/usr/bin/env python3
"""Final paper figure: GRPO training curves, 3 vertically stacked panels, shared x.

(a) mean training reward, (b) held-out pass@1 (official test split), (c) cumulative
FLOPs consumed — embedding-inclusive at policy-model parity (matches tab:grpo).
Color groups by support type (baseline / rolling / permanent); linestyle encodes
theta; linewidth encodes m_c. Writes paper/figures/fig_grpo_training_curves.{pdf,png}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "notebooks/data/tau2_es/rollout_cache"
FIGS = ROOT / "paper/figures"
MATCHED = 50
N_PARAMS = 8.19e9
EMBED_PRICE = 0.13e-6            # $/token (EmbeddingClient.estimated_cost constant)
PRICE_TRAIN = json.load(open(ROOT / "notebooks/data/tau2_es/calibration.json"))["price_train_per_tok"]

# variant -> (runs, support, theta, m_c);  dirs resolved against logs/ then logs_chpc/
VARIANTS = {
    "baseline":  dict(dirs=[f"tau2_baseline_retail_official_s{s}" for s in (1, 2, 3)],
                      support="baseline", theta=None, mc=None, label="GRPO baseline"),
    "es08":      dict(dirs=[f"tau2_early_stop_retail_official_s{s}" for s in (1, 2, 3)],
                      support="rolling", theta=0.8, mc=4,
                      label=r"rolling $\theta$=0.8 $m_c$=4"),
    "es07":      dict(dirs=[f"tau2_early_stop_retail_official_s{s}_t07" for s in (1, 2, 3)],
                      support="rolling", theta=0.7, mc=4,
                      label=r"rolling $\theta$=0.7 $m_c$=4"),
    "es07_mc0":  dict(dirs=[f"tau2_early_stop_retail_official_s{s}_mc0_t07" for s in (1, 2, 3)],
                      support="rolling", theta=0.7, mc=0,
                      label=r"rolling $\theta$=0.7 $m_c$=0"),
    "es065_mc0": dict(dirs=[f"tau2_early_stop_retail_official_s{s}_mc0_t065" for s in (1, 2, 3)],
                      support="rolling", theta=0.65, mc=0,
                      label=r"rolling $\theta$=0.65 $m_c$=0"),
    "limit":     dict(dirs=[f"tau2_best_avail_strat_mc0_retail_official_s{s}" for s in (1, 2, 3)],
                      support="permanent", theta=0.8, mc=0,
                      label=r"permanent $\theta$=0.8 $m_c$=0"),
    "limit07":   dict(dirs=[f"tau2_best_avail_strat_mc0_retail_official_s{s}_t07" for s in (1, 2, 3)],
                      support="permanent", theta=0.7, mc=0,
                      label=r"permanent $\theta$=0.7 $m_c$=0"),
}
# rollout-cache parquet names by (variant, seed index)
CACHE_NAME = {
    ("baseline", i): f"baseline_s{i}" for i in (1, 2, 3)
} | {("es08", i): f"es08_s{i}" for i in (1, 2, 3)} \
  | {("es07", i): f"es07_s{i}" for i in (1, 2, 3)} \
  | {("es07_mc0", i): f"es07_mc0_s{i}" for i in (1, 2, 3)} \
  | {("es065_mc0", i): f"es065_mc0_s{i}" for i in (1, 2, 3)} \
  | {("limit", i): f"limit_s{i}" for i in (1, 2, 3)} \
  | {("limit07", i): f"limit07_s{i}" for i in (1, 2, 3)}

COLOR = {"baseline": "#6b7280", "rolling": "#0072B2", "permanent": "#E69F00"}
LS = {None: "-", 0.8: "-", 0.7: "--", 0.65: ":"}
LW = {None: 2.2, 4: 1.9, 0: 1.3}

def find_run_dir(d):
    for base in ("logs", "logs_chpc"):
        p = ROOT / base / d
        if (p / "metrics.jsonl").exists():
            return p
    raise FileNotFoundError(d)

def deltas(vals):
    """Cumulative-within-session field -> per-batch deltas (value drop = restart)."""
    out, prev = [], 0.0
    for v in vals:
        v = v or 0.0
        out.append(v - prev if v >= prev else v)
        prev = v
    return out

def load_run(variant, i):
    """Per-batch frame: reward, pass_rate (sparse), total_flops incl. embedding."""
    vc = VARIANTS[variant]
    md = find_run_dir(vc["dirs"][i - 1])
    M = {r["batch"]: r for l in open(md / "metrics.jsonl") if (r := json.loads(l))}
    batches = sorted(b for b in M if b < MATCHED)
    train_tok = np.array(deltas([M[b].get("cost_train") for b in batches])) / PRICE_TRAIN
    embed_usd = np.array(deltas([(M[b].get("early_stop") or {}).get("embedding_cost_usd")
                                 for b in batches]))
    emb_tok = embed_usd / EMBED_PRICE
    pq = pd.read_parquet(CACHE / f"{CACHE_NAME[(variant, i)]}.parquet")
    gen = (pq[pq.batch < MATCHED].groupby("batch")
             .apply(lambda g: (g.prefill_tokens + g.sample_tokens).sum(), include_groups=False)
             .reindex(batches).fillna(0).values)
    flops = 2 * N_PARAMS * gen + 6 * N_PARAMS * train_tok + 2 * N_PARAMS * emb_tok
    return pd.DataFrame({
        "batch": batches,
        "reward": [M[b].get("avg_reward_before_filter") for b in batches],
        "pass_rate": [(M[b].get("eval") or {}).get("pass_rate") for b in batches],
        "flops": flops,
    })

data = {}
for v in VARIANTS:
    runs = []
    for i in (1, 2, 3):
        try:
            runs.append(load_run(v, i))
        except FileNotFoundError as e:
            print(f"  [missing] {v} seed {i}: {e}")
    data[v] = runs
    print(f"{v}: {len(runs)} seeds, {sum(len(r) for r in runs)} batch rows")

# pass_rate scale check
pr = pd.concat([r for rs in data.values() for r in rs]).pass_rate.dropna()
SCALE = 100 if pr.max() <= 1.0 else 1

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.6), sharex=True,
                         gridspec_kw=dict(hspace=0.28))

def style(v):
    vc = VARIANTS[v]
    return dict(color=COLOR[vc["support"]], ls=LS[vc["theta"]], lw=LW[vc["mc"]])

# (a) training reward (rolling mean 5, mean +/- sem over seeds)
ax = axes[0]
for v, runs in data.items():
    arr = pd.concat([r.set_index("batch").reward for r in runs], axis=1)
    m = arr.mean(1).rolling(5, min_periods=1).mean()
    ax.plot(m.index, m.values, **style(v), label=VARIANTS[v]["label"])
    if arr.shape[1] > 1:
        sem = (arr.std(1) / np.sqrt(arr.shape[1])).rolling(5, min_periods=1).mean()
        ax.fill_between(m.index, m - sem, m + sem, color=COLOR[VARIANTS[v]["support"]],
                        alpha=0.10, lw=0)
ax.set_ylabel("Mean training reward")
ax.set_title("(a) Training reward", loc="left", fontsize=9)

# (b) held-out pass@1 (evals every 5 batches; mean, min-max band)
ax = axes[1]
for v, runs in data.items():
    arr = pd.concat([r.set_index("batch").pass_rate for r in runs], axis=1).dropna(how="all")
    m = arr.mean(1) * SCALE
    ax.plot(m.index, m.values, marker="o", ms=2.6, **style(v))
    if arr.shape[1] > 1:
        ax.fill_between(m.index, arr.min(1) * SCALE, arr.max(1) * SCALE,
                        color=COLOR[VARIANTS[v]["support"]], alpha=0.10, lw=0)
ax.set_ylabel("Held-out pass@1 (%)")
ax.set_title("(b) Test-split pass@1", loc="left", fontsize=9)

# (c) cumulative FLOPs incl. embedding at parity; baseline is the top line
ax = axes[2]
ends = []
for v, runs in data.items():
    curves = [np.cumsum(r.flops.values) / 1e18 for r in runs]
    L = min(map(len, curves))
    arr = np.stack([c[:L] for c in curves])
    x = np.arange(L)
    ax.plot(x, arr.mean(0), **style(v))
    if len(curves) > 1:
        ax.fill_between(x, arr.min(0), arr.max(0), color=COLOR[VARIANTS[v]["support"]],
                        alpha=0.10, lw=0)
    ends.append((v, arr.mean(0)[-1]))
for v, y in ends:                     # right-edge % saved annotations
    base = dict(ends)["baseline"]
    if v != "baseline":
        ax.annotate(f"–{100 * (1 - y / base):.0f}%", (MATCHED - 1, y),
                    xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                    color=COLOR[VARIANTS[v]["support"]], va="center")
ax.set_ylabel("Cumulative FLOPs (×10¹⁸)\nincl. embedding")
ax.set_title("(c) Compute consumed", loc="left", fontsize=9)
ax.set_xlabel("Training batch")
ax.set_xlim(0, MATCHED + 4.5)

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(0.54, 1.005), ncol=2, fontsize=6.4, frameon=False,
           handlelength=2.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.925))
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "fig_grpo_training_curves.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_grpo_training_curves.png", dpi=200, bbox_inches="tight")
print("saved", FIGS / "fig_grpo_training_curves.{pdf,png}")

# ---- Table 3 savings column (embedding-inclusive at policy parity) ----
totals = {v: np.array([r.flops.sum() for r in runs]) / 1e18
          for v, runs in data.items() if runs}
base = totals["baseline"].mean()
print(f"\nbaseline total (incl-embed): {base:.1f} ± {totals['baseline'].std():.1f} EFLOPs")
print(f"{'variant':12} {'incl-embed EF':>14}   {'saved % (Table 3)':>18}")
for v, t in totals.items():
    if v == "baseline":
        continue
    sv = 100 * (1 - t / base)
    print(f"{v:12} {t.mean():9.1f}±{t.std():.1f}   {sv.mean():12.1f} ± {sv.std():.1f}")
