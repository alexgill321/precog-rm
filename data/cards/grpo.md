---
license: mit
tags: [llm-agents, grpo, reinforcement-learning, early-stopping, tau2-bench]
pretty_name: Precog-RM GRPO training logs
size_categories: [10B<n<100B]
---

# Precog-RM GRPO training logs

The 21 official GRPO runs behind Section 5 of *"Divergent Paths: Learning an
Outcome-Predictive Representation of LLM Agent Trajectories"* (Gill, Gogia, Guzman):
Qwen3-8B (LoRA r=32) on the official tau2-bench retail train split, group size 16,
20 tasks per batch, three seeds per configuration, with and without the Precog-RM
early-stopping controller. Code: https://github.com/alexgill321/precog-rm

```
runs/<run_name>/metrics.jsonl        per-batch metrics (rewards, billed costs, early-stop stats,
                                     "eval" rows with test-split pass rate every 5 batches)
runs/<run_name>/rollouts/batch_<b>.jsonl   full rollout records: conversation, per-turn prefill/sample
                                     token counts, PRM score trajectory, early-stop decision
runs/<run_name>/support_sets/        controller support sets (early-stop runs)
replay/results_<config>_s<seed>.json simulated early stopping replayed over the baseline seeds
eval/eval_base_qwen3_8b_test.json    base model on the test split (40 tasks x 3 samples)
```

Runs: `tau2_baseline_retail_official_s{1,2,3}` (100 batches; the paper's matched window
is batches 0-49), `tau2_early_stop_retail_official_s{1,2,3}` (rolling support, theta 0.8,
m_c 4), `..._t07` (theta 0.7), `..._mc0_t07`, `..._mc0_t065` (m_c 0),
`tau2_best_avail_strat_mc0_retail_official_s{1,2,3}` (permanent support, theta 0.8, m_c 0),
`..._t07` (theta 0.7).

Reproduce Table 3 / Figure 5 / Appendix K: `scripts/grpo_cache.py` (per-rollout token
cache), `scripts/table_grpo.py`, `scripts/fig_grpo_curves.py`; Appendix M:
`scripts/replay_sweep_table.py`, `scripts/fig_sim_pareto.py`.
