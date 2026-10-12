"""Shared definitions for the GRPO early-stopping analysis (Section 5, App. K).

Run registry (the 21 official runs), the FLOPs accounting of Appendix K, and a
per-run loader that joins `metrics.jsonl` (billed cost deltas -> trained and
embedded tokens, in-training eval pass rates) with the per-rollout token cache
(`notebooks/data/tau2_es/rollout_cache/<run>.parquet`, built from the raw
rollout logs by scripts/grpo_cache.py).

FLOPs = 2N * T_gen + 6N * T_train + 2N_emb * T_emb, N = 8.19e9 (Qwen3-8B);
the paper charges the controller at policy parity (N_emb = N).
"""
import json

import numpy as np
import pandas as pd

from _paths import REPO, RUNS, GRPO_CACHE, EVAL

MATCHED = 50                 # matched comparison window: batches 0..49
N_PARAMS = 8.19e9
EMBED_PRICE = 0.13e-6        # $/token, the constant in EmbeddingClient.estimated_cost
EMBED_PARITY_PARAMS = 8.19e9
CACHE_DIR = GRPO_CACHE / "rollout_cache"
CALIBRATION = GRPO_CACHE / "calibration.json"

VARIANTS = {
    "baseline":  dict(dirs=[f"tau2_baseline_retail_official_s{s}" for s in (1, 2, 3)],
                      support="baseline", theta=None, mc=None, label="GRPO baseline"),
    "es08":      dict(dirs=[f"tau2_early_stop_retail_official_s{s}" for s in (1, 2, 3)],
                      support="rolling", theta=0.8, mc=4, label=r"roll $\theta$=.8 $m_c$=4"),
    "es07":      dict(dirs=[f"tau2_early_stop_retail_official_s{s}_t07" for s in (1, 2, 3)],
                      support="rolling", theta=0.7, mc=4, label=r"roll $\theta$=.7 $m_c$=4"),
    "es07_mc0":  dict(dirs=[f"tau2_early_stop_retail_official_s{s}_mc0_t07" for s in (1, 2, 3)],
                      support="rolling", theta=0.7, mc=0, label=r"roll $\theta$=.7 $m_c$=0"),
    "es065_mc0": dict(dirs=[f"tau2_early_stop_retail_official_s{s}_mc0_t065" for s in (1, 2, 3)],
                      support="rolling", theta=0.65, mc=0, label=r"roll $\theta$=.65 $m_c$=0"),
    "limit":     dict(dirs=[f"tau2_best_avail_strat_mc0_retail_official_s{s}" for s in (1, 2, 3)],
                      support="permanent", theta=0.8, mc=0, label=r"perm $\theta$=.8 $m_c$=0"),
    "limit07":   dict(dirs=[f"tau2_best_avail_strat_mc0_retail_official_s{s}_t07" for s in (1, 2, 3)],
                      support="permanent", theta=0.7, mc=0, label=r"perm $\theta$=.7 $m_c$=0"),
}
SEEDS = (1, 2, 3)


def cache_name(variant, seed):
    return f"{variant}_s{seed}"


def run_dir(variant, seed):
    """Run directory: the data release ($PRECOG_DATA/runs) or, for metrics-only
    use, the copies of metrics.jsonl shipped in the repo under logs/."""
    name = VARIANTS[variant]["dirs"][seed - 1]
    for base in (RUNS, REPO / "logs"):
        if (base / name / "metrics.jsonl").exists():
            return base / name
    raise FileNotFoundError(f"{name}/metrics.jsonl (download the grpo stage or set PRECOG_DATA)")


def deltas(vals):
    """Cumulative-within-session field -> per-batch deltas (a drop = session restart)."""
    out, prev = [], 0.0
    for v in vals:
        v = v or 0.0
        out.append(v - prev if v >= prev else v)
        prev = v
    return out


def price_train():
    return json.load(open(CALIBRATION))["price_train_per_tok"]


def load_metrics(variant, seed):
    return {r["batch"]: r for l in open(run_dir(variant, seed) / "metrics.jsonl")
            if l.strip() and (r := json.loads(l))}


def load_run(variant, seed, matched=MATCHED):
    """Per-batch frame for batches < matched: reward, pass_rate (sparse, in %),
    gen/train/emb token counts and the embedding-inclusive FLOPs."""
    M = load_metrics(variant, seed)
    batches = sorted(b for b in M if b < matched)
    train_tok = np.array(deltas([M[b].get("cost_train") for b in batches])) / price_train()
    emb_tok = np.array(deltas([(M[b].get("early_stop") or {}).get("embedding_cost_usd")
                               for b in batches])) / EMBED_PRICE
    pq = pd.read_parquet(CACHE_DIR / f"{cache_name(variant, seed)}.parquet")
    gen = (pq[pq.batch < matched].groupby("batch")
             .apply(lambda g: (g.prefill_tokens + g.sample_tokens).sum(), include_groups=False)
             .reindex(batches).fillna(0).values)
    pr = np.array([(M[b].get("eval") or {}).get("pass_rate", np.nan) for b in batches], dtype=float)
    if np.nanmax(pr) <= 1.0:
        pr = pr * 100
    return pd.DataFrame({
        "batch": batches,
        "reward": [M[b].get("avg_reward_before_filter") for b in batches],
        "pass_rate": pr,
        "gen_tok": gen, "train_tok": train_tok, "emb_tok": emb_tok,
        "flops": 2 * N_PARAMS * gen + 6 * N_PARAMS * train_tok + 2 * EMBED_PARITY_PARAMS * emb_tok,
        "flops_excl_emb": 2 * N_PARAMS * gen + 6 * N_PARAMS * train_tok,
    })


def load_all(verbose=True):
    data = {}
    for v in VARIANTS:
        runs = []
        for s in SEEDS:
            try:
                runs.append(load_run(v, s))
            except FileNotFoundError as e:
                print(f"  [missing] {v} seed {s}: {e}")
        data[v] = runs
        if verbose:
            print(f"{v}: {len(runs)} seeds, {sum(len(r) for r in runs)} batch rows")
    return data


def base_model_pass_rate():
    for p in (EVAL / "eval_base_qwen3_8b_test.json", REPO / "exp/results/grpo/eval_base_qwen3_8b_test.json"):
        if p.exists():
            return float(json.load(open(p))["metrics"]["pass_rate"])
    return None
