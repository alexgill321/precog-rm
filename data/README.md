# Data root layout

Everything large lives under one data root, `$PRECOG_DATA` (default: this
directory). `python scripts/download.py --part ...` fills it from the
HuggingFace release (dataset `agill32/precog-rm`, model `agill32/precog-rm-checkpoints`)
and then builds the `table1_data/` view the loaders read. Nothing here is
needed for the "shipped results" fast path (`print_table1.py`,
`aggregate_generalization.py --check`, `table_grpo.py`, ... read small JSONs
and caches that live in the repo).

Benchmarks: `tau2`, `bfclv3`, `agentclinic`. Policy folders: `gpt5mini`,
`qwen3-80b`, `deepseek-v3.2` (the code's internal keys are `gpt5mini`,
`qwen`, `deepseek`; `scripts/_paths.py:POLICY_DIR` maps them).

```
$PRECOG_DATA/
  rollouts/<benchmark>/<policy>/<task>/      raw trajectories (public layout, one JSON per rollout)
      rollout_NNN.json                       envelope: benchmark, policy, task, rid, reward, n_steps,
                                             steps=[{action, observation}] (embedding order), raw={harness record}
      extra_rollouts.jsonl                   surplus rollouts beyond the 50 the paper used (tau2/qwen3-80b)
      task.json                              task id, category/domain, success rate, source run
  embeddings/<benchmark>/<policy>/<task>/    action.npz, observation.npz (keys r<rid>_s<step>), rewards.json
  embeddings/reasoning/tau2_qwen_steps/      per-step reasoning embeddings (App. B.1)
  embeddings/grpo_replay/official_retail_s{1,2,3}/   baseline-GRPO rollouts embedded per batch (App. M)
  grpo/runs/<run_name>/                      21 official GRPO runs: metrics.jsonl, rollouts/batch_*.jsonl, support_sets/
  grpo/replay/results_<config>_s<seed>.json  simulated early stopping
  grpo/eval/eval_base_qwen3_8b_test.json     base model on the official test split
  caches/mmap/, caches/task_splits.json      optional float16 fast path (what every experiment reads)
  ablations/agentclinic/notools/<policy>/    no-tools ablation rollouts (25 per scenario)
  ablations/agentclinic/resample/<policy>.json   final-diagnosis resamples
  ablations/agentclinic/terra/               GPT-5.6-Terra reference rollouts
  meta/                                      tau2_task_ids.json, tau2_telecom_id_mapping.json,
                                             bfclv3_mixed_200_tasks.json, toolreliant_*.txt, nontool_*.txt
  ckpts/precog/, ckpts/ablations/, ckpts/prefix/   checkpoints (download.py re-folders the model repo)

  table1_data/<benchmark>_<model>/           DERIVED by download.py --materialize-only from embeddings/:
      embeddings/{action,response}/<task>.npz    hard links
      <category>/<id>_tdp_<rid>.json             reward stubs
      manifest.json                              tau2_qwen only
```

Run names: `tau2_baseline_retail_official_s{1,2,3}` (100 batches),
`tau2_early_stop_retail_official_s{1,2,3}[_t07][_mc0_t07][_mc0_t065]`,
`tau2_best_avail_strat_mc0_retail_official_s{1,2,3}[_t07]` (50 batches).

Provenance of each cell (verified against the caches the paper's results were
computed from) is in `docs/HF_LAYOUT_PLAN.md` and in every `task.json`. Two
tau2/qwen3-80b tasks (`retail_31`, `retail_42`) ship rollouts but no embeddings:
their source embeddings had drifted from the rollout files and neither task
enters any result (both are degenerate under the 50-rollout subsample).
