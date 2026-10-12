# Data root layout

Everything large lives under one data root, `$PRECOG_DATA` (default: this
directory). Fill it with `python scripts/download.py --stage ...` or point
`PRECOG_DATA` at a directory with this layout. Each stage is an entry point
into the pipeline; nothing here is required for the "shipped results" fast
path (`print_table1.py`, `aggregate_generalization.py --check`, `table_grpo.py`
read small JSONs and caches that live in the repo).

```
$PRECOG_DATA/
  table1_data/<benchmark>_<model>/        9 embedding corpora (layer 1, ~25 GB total)
      benchmark in {tau2, bfclv3, agentclinic}, model in {gpt5mini, qwen, deepseek}
      embeddings/action/<task>.npz        keys r<rid>_s<step>, float32 (3072,)  text-embedding-3-large
      embeddings/response/<task>.npz      same keys; observation embeddings
      <category>/<task>_tdp_<rid>.json    reward stubs (bfclv3, agentclinic)
      <domain>/<idx>/rollout_<rid>.json   FULL rollouts incl. messages (tau2 cells)
      manifest.json                       tau2_qwen only: per-rollout rewards/sources (consolidated set)
  reasoning_emb/tau2_qwen_steps/<task>.npz   per-step reasoning embeddings, tau2/qwen test tasks (App. B.1)
  grpo_replay/official_retail_s{1,2,3}/   baseline-GRPO rollouts embedded per batch-task (replay simulation)
      data/batch_<b>_retail_<task>.json   per-turn prefill token counts, rewards
      embeddings/{action,response}/*.npz
  caches/
      slim/<benchmark>_<model>.pkl        intermediate (optional; built by build_slim_caches.py)
      mmap/<benchmark>_<model>_{act,res,pooled}.npy + _meta.pkl   what every experiment reads
      task_splits.json                    (copy; the canonical file is exp/results/protonet_ablations/)
  ckpts/
      precog/precog_<pool>_s<seed>.pt     75 sweep checkpoints: <benchmark>_<g|q|d combos>, two- and
                                          three-benchmark pools as <b1>-<b2>[-<b3>]_all; seeds 0-2
      ablations/src_<name>_s0.pt, mix_<name>_s0.pt, pool_<pool>_full_s0.pt   design study (App. C/F)
      prefix/prefix_<pool>_<variant>_s0.pt                                    prefix-trained encoders (Sec. 4.5)
      prefix/grpo_controller_model_prefix.pt                                  the controller deployed in GRPO
  runs/<run_name>/                        21 official GRPO runs (~35 GB)
      metrics.jsonl                       per-batch metrics incl. "eval" rows every 5 batches
      rollouts/batch_<b>.jsonl            full rollout records (conversation, tokens, PRM scores)
      support_sets/                       early-stop runs only
  replay/results_<config>_s<seed>.json    simulated early stopping over the baseline seeds (App. M)
  eval/eval_base_qwen3_8b_test.json       base model on the official test split (20.8%)
  rollouts/
      raw/bfclv3/<model>/<category>/<task>/rollout_<rid>/{result,score}/   BFCL runs
      raw/agentclinic/<model>/scenario_<sid>_rollout_<rid>.json           MedQA-Ext, 50 rollouts
      raw/agentclinic/gpt56terra/                                         9 scenarios x 50 (App. I)
      raw/agentclinic_notools/<model>/                                    no-tools ablation, 25 rollouts
      resample/<model>.json                                               final-diagnosis resamples (App. I)
      meta/toolreliant_*.txt, nontool_*.txt, tau2_task_ids.json, bfclv3_mixed_200_tasks.json
```

Run names: `tau2_baseline_retail_official_s{1,2,3}` (100 batches),
`tau2_early_stop_retail_official_s{1,2,3}[_t07][_mc0_t07][_mc0_t065]`,
`tau2_best_avail_strat_mc0_retail_official_s{1,2,3}[_t07]` (50 batches).

Raw tau2 rollouts for the three policies are inside their `table1_data` cells
(the tau2 corpora were never repacked into stubs). The tau2/qwen cell holds
more than 50 rollouts for many tasks; every loader caps at the first 50.
