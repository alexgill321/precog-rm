#!/usr/bin/env python3
"""Table 3 (tab:grpo), Appendix K's token table (tab:flops-detail), and the
training-trend numbers of Section 5.2, from the 21 official GRPO runs.

Inputs: $PRECOG_DATA/runs/<run>/metrics.jsonl, the per-rollout token cache
(notebooks/data/tau2_es/rollout_cache, rebuilt by scripts/grpo_cache.py), and
the base-model eval $PRECOG_DATA/eval/eval_base_qwen3_8b_test.json.

    python scripts/table_grpo.py
"""
import numpy as np

from grpo_common import VARIANTS, MATCHED, load_all, base_model_pass_rate, N_PARAMS, EMBED_PARITY_PARAMS


def ms(x, f="{:.1f}"):
    x = np.asarray(x, dtype=float)
    return f"{f.format(x.mean())} ± {f.format(x.std())}" if len(x) > 1 else f.format(x.mean())


def main():
    data = load_all()
    base_pass = base_model_pass_rate()

    # ---- per-run totals over the matched window ----
    tot = {v: dict(
        flops=np.array([r.flops.sum() for r in runs]),
        flops_excl=np.array([r.flops_excl_emb.sum() for r in runs]),
        gen=np.array([r.gen_tok.sum() for r in runs]),
        train=np.array([r.train_tok.sum() for r in runs]),
        emb=np.array([r.emb_tok.sum() for r in runs]),
        last_eval=np.array([r.dropna(subset=["pass_rate"]).iloc[-1].pass_rate for r in runs]),
        last_eval_batch=[int(r.dropna(subset=["pass_rate"]).iloc[-1].batch) for r in runs],
        late_reward=np.array([r[(r.batch >= MATCHED - 10)].reward.mean() for r in runs]),
    ) for v, runs in data.items() if runs}
    base = tot["baseline"]["flops"].mean()
    base_pass_grpo = tot["baseline"]["last_eval"].mean()

    print(f"\n=== Table 3: GRPO on tau2-retail with Precog-RM early stopping "
          f"(50 batches, Qwen3-8B; mean ± std over 3 seeds, std ddof=0) ===")
    print(f"pass@1 evaluated at batch {sorted(set(tot['baseline']['last_eval_batch']))} "
          f"(last eval inside the matched window); base Qwen3-8B = {base_pass if base_pass is None else round(base_pass, 1)}%")
    print(f"{'support':10} {'theta':>5} {'m_c':>3} {'FLOPs saved (%)':>16} {'Pass@1':>13} {'retained gain':>13}")
    for v, vc in VARIANTS.items():
        if v not in tot:
            continue
        t = tot[v]
        saved = 100 * (1 - t["flops"] / base)
        gain = ((t["last_eval"].mean() - base_pass) / (base_pass_grpo - base_pass)
                if base_pass is not None and v != "baseline" else np.nan)
        print(f"{vc['support']:10} {str(vc['theta']):>5} {str(vc['mc']):>3} "
              f"{ms(saved) if v != 'baseline' else '---':>16} {ms(t['last_eval']):>13} "
              f"{'' if np.isnan(gain) else f'{gain:.0%}':>13}")

    print(f"\n=== Appendix K (tab:flops-detail): tokens in billions, matched window; "
          f"savings charge the controller at policy parity ===")
    print(f"baseline total (incl. embedding): {ms(tot['baseline']['flops'] / 1e18)} EFLOPs")
    print(f"{'config':24} {'gen (G)':>8} {'train (G)':>9} {'emb (G)':>8} {'emb/gen':>8} "
          f"{'saved excl (%)':>14} {'saved incl (%)':>14}")
    for v, vc in VARIANTS.items():
        if v not in tot:
            continue
        t = tot[v]
        excl = 100 * (1 - t["flops_excl"] / base)
        incl = 100 * (1 - t["flops"] / base)
        print(f"{vc['support'] + ' θ=' + str(vc['theta']) + ' mc=' + str(vc['mc']):24} "
              f"{t['gen'].mean() / 1e9:8.2f} {t['train'].mean() / 1e9:9.3f} "
              f"{t['emb'].mean() / 1e9:8.3f} {np.mean(t['emb'] / t['gen']):8.1%} "
              f"{ms(excl) if v != 'baseline' else '---':>14} {ms(incl) if v != 'baseline' else '---':>14}")
    g, tr = tot["baseline"]["gen"], tot["baseline"]["train"]
    print(f"baseline FLOPs share: generation {np.mean(2 * g / (2 * g + 6 * tr)):.0%}, "
          f"training {np.mean(6 * tr / (2 * g + 6 * tr)):.0%}")

    print("\n=== Training trends: mean assigned reward, batches 40-49 ===")
    for v, vc in VARIANTS.items():
        if v in tot:
            print(f"  {vc['label']:28} {ms(tot[v]['late_reward'], '{:.3f}')}")


if __name__ == "__main__":
    main()
