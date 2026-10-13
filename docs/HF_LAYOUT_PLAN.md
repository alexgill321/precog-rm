# Plan: restructure the HuggingFace release

Goal: one dataset repo a reader can browse top-down, with four clearly
separated parts, and every file traceable to the exact source the paper's
results were computed from.

## Sources of truth (verified 2026-10-13)

The authoritative cell -> directory map is the `SOURCES` dict in
`notebooks/protonet_ablations.ipynb` (cell 1), which `build_slim_caches.py`,
`merged_cv_table.py`, the sweep, and every table script read through the slim
and mmap caches. Each embedded cell was checked against its raw run: task sets
identical, rollout ids identical, no mismatches.

| Cell | Raw rollouts (text) | Embeddings (what results load) | Notes |
|---|---|---|---|
| tau2 / GPT-5-mini | `exp/models/gpt5mini_200/data` | same dir, `embeddings/` | 200 tasks x exactly 50 |
| tau2 / DeepSeek-V3.2 | `exp/models/dsv32_200/data` | same dir, `embeddings/` | 200 tasks x exactly 50 |
| tau2 / Qwen3-80B | `exp/tau2/baseline/consolidated_baselines/data` (+ `manifest.json` with per-rollout provenance from 8 earlier runs) | same dir, `embeddings/` | 200 tasks; 155 tasks have >50 rollouts (up to 589) |
| bfclv3 / GPT-5-mini | `exp/bfcl/runs/gpt5mini_repro_200x50` | `exp/bfcl/table1_data_gpt5mini` (bundled-step embeddings, v3) | 200 tasks x 50 |
| bfclv3 / Qwen3-80B | `exp/bfcl/runs/400_across_cats` (400 task dirs; 200 are the paper set) | `exp/bfcl/table1_data_qwen_v2` (per-call steps) | ship only the 200 preset tasks |
| bfclv3 / DeepSeek-V3.2 | `exp/bfcl/runs/dsv32_194_prompt` (201 task dirs) | `exp/bfcl/table1_data_dsv32_v2` (per-call steps) | ship only the 200 preset tasks |
| agentclinic / GPT-5-mini | `benchmark_refs/AgentClinic/outputs/medqa-ext-gpt5mini-50rollouts` | `exp/agentclinic/table1_data_gpt5mini_medqa_v2` | 214 scenarios x 50 |
| agentclinic / Qwen3-80B | `.../medqa-ext-qwen-50rollouts` | `.../table1_data_qwen_medqa_v2` | 214 x 50 |
| agentclinic / DeepSeek-V3.2 | `.../medqa-ext-deepseek-10rollouts` (expanded in place to 50) | `.../table1_data_deepseek_medqa_v2` | 214 x 50 |

Not shipped (old or unused): tau2 `saturation/*`, `neardegenerate/*`,
`dsv32_190`, `dsv32_scout`, OLMo runs; BFCL `table1_data` (v1), `_v1`, `_v3`
variants and the 200 screen-only qwen tasks; AgentClinic NEJM runs, `_nodiag`
and `_v1` embedding variants (regenerable from the shipped files).

### The 50-rollout rule

Every loader calls `load_all_tasks(max_rollouts=50)`, which for a task with
more than 50 rollouts keeps a seeded random subset
(`np.random.RandomState(subsample_seed).choice(sorted(rids), 50)`). This only
affects tau2 / Qwen3-80B. The release must split those tasks with the same
rule, so `rollouts/` holds exactly the 50 the paper used and
`extra_rollouts/` the rest. The split script reuses the loader function to
guarantee identity, and the mmap cache's rollout ids are the acceptance test.

## Target layout: one dataset repo `agill32/precog-rm`

```
README.md                                   what each part is, how to download one part, how the code consumes it

rollouts/<benchmark>/<policy>/<task>/
    rollout_000.json ... rollout_049.json   one JSON per rollout, common envelope (below)
    extra_rollouts/rollout_NNN.json         only tau2/qwen tasks with >50 (ids keep their original rid)
    task.json                               task id, benchmark category/domain, success rate over the 50
  benchmark in {tau2, bfclv3, agentclinic}; policy in {gpt5mini, qwen3-80b, deepseek-v3.2}
  task = airline_12 | retail_103 | telecom_7 (tau2, with the original long telecom id inside task.json),
         base_0 | longctx_15 | missfunc_42 | missparam_7 (bfclv3), scenario_0013 (agentclinic)

embeddings/<benchmark>/<policy>/<task>/
    action.npz      keys r<rid>_s<step>, float32 (3072,)     exactly the rollouts in rollouts/<...>/<task>/ (not extra)
    observation.npz same keys
    rewards.json    {rid: reward}
  plus embeddings/reasoning/tau2/qwen3-80b/<task>.npz      per-step reasoning embeddings (App. B.1)
  plus embeddings/grpo_replay/official_retail_s{1,2,3}/    baseline-GRPO rollouts embedded per batch (App. M)

grpo/
    runs/<run_name>/metrics.jsonl, rollouts/batch_NNNNNN.jsonl, support_sets/     21 official runs (unchanged)
    replay/results_<config>_s<seed>.json                                           simulated early stopping
    eval/eval_base_qwen3_8b_test.json                                              base-model reference

caches/mmap/   optional fast path (float16 mmap caches + task_splits.json), documented as derived
```

Model repo `agill32/precog-rm-checkpoints`, re-foldered for clarity:

```
sweep/precog_<pool>_s<seed>.pt            75 main-sweep checkpoints (Tables 1-2, Fig 3, App J)
design_study/src_<name>_s0.pt, mix_<name>_s0.pt, pool_<pool>_full_s0.pt     App C/F
prefix/prefix_<pool>_<variant>_s0.pt      prefix-trained encoders (Sec 4.5)
grpo_controller.pt                        the controller deployed inside GRPO (Sec 5)
```

The current four repos (`precog-rm-embeddings`, `-grpo-logs`, `-rollouts`,
`-checkpoints`) are deleted after the new ones verify; the collection then
points at the two new repos.

### Rollout envelope (one JSON per rollout, all benchmarks)

```
{ "benchmark": "tau2", "policy": "qwen3-80b", "task": "retail_103", "rid": 7,
  "reward": 1.0, "n_steps": 23,
  "steps": [{"action": "...", "observation": "..."}, ...],     # exactly the texts that were embedded
  "raw": {...} }                                                # the untouched harness record
```

`steps` makes the embedding correspondence explicit (step t of `steps` is
`r<rid>_s<t>` in the npz files); `raw` keeps the full original record: tau2's
messages/steps/reward_info, BFCL's gorilla `result` and `score` outputs, and
AgentClinic's history/tokens/cost. BFCL's current on-disk form (a directory
tree of 60+ files per rollout) is collapsed into this one JSON.

### Browsability vs. Hub limits

Per-rollout files: about 30k (tau2) + 30k (bfcl) + 32k (agentclinic) + 24k
extra-rollouts = ~116k small files. The Hub recommends staying under 100k files
per repo and 10k per folder (per-task folders are far below that), and commits
hold ~75 small files with 128 commits per hour, so the upload takes roughly
12-15 hours of wall clock in resumable background runs. If that is too slow or
too close to the limit, the fallback is one `rollouts.jsonl` per task (50
records; 600-1,000 files total), which still lets a reader open a task and page
through its 50 rollouts but loses per-file browsing. Default: per-rollout files
for the 50 paper rollouts, and a single `extra_rollouts.jsonl` per task for the
surplus tau2/qwen rollouts (the 24k files nobody needs to browse one by one).
That lands at ~92k files.

## Code changes

1. `scripts/build_release.py` (new): builds the target tree from the local
   sources above into a staging dir: per-rollout envelopes with extracted
   `steps` (reusing `src/prm/embeddings.extract_sample_text/extract_response_text`,
   the BFCL parsers in `generation/bfcl/`, and the AgentClinic extractor), the
   50/extra split via the loader's own subsampling, per-task embedding folders
   copied from the cells, and `task.json`. Runs the secret scan at the end.
2. `scripts/verify_release.py` (new): for every cell, asserts rollout ids and
   step counts match between `rollouts/` and `embeddings/`, that the shipped
   50 equal the ids in the mmap caches, that BFCL has exactly the 200 preset
   tasks and AgentClinic 214 scenarios, and that no credential-like content
   exists. This is the gate before upload.
3. `scripts/download.py`: download by part (`--part rollouts|embeddings|grpo|caches|checkpoints`,
   with `--benchmark/--policy` filters), and materialize the layout the code
   expects (`table1_data/<benchmark>_<policy>/` with `embeddings/{action,response}/<task>.npz`
   plus reward stubs) with hard links or symlinks, so no loader changes.
4. `scripts/upload_hf.py`: target the new repos; keep the exclusion list and
   secret scan; add `--part` to upload incrementally and resume.
5. Docs: `data/README.md` and the dataset card rewritten around the four
   parts; `REPLICATION.md`/`GENERATION.md` paths updated; the paper's Data
   link stays on the collection.

## Execution order

1. `build_release.py` for tau2 (all three policies) -> verify -> upload `rollouts/tau2`, `embeddings/tau2`.
2. Same for bfclv3 (needs the BFCL per-rollout collapse) and agentclinic.
3. Move grpo and caches into the new repo (already clean; re-upload from staging).
4. Re-folder checkpoints (small).
5. Fresh-clone test: download one benchmark, run `build_slim_caches.py` on it,
   and check the mmap cache rollout ids match the shipped ones.
6. Delete the old repos, update the collection, flip public.

## Open decisions for you

- Per-rollout files for the 50 (recommended, ~12-15 h upload) vs one JSONL per task.
- Keep the 200 screen-only BFCL qwen tasks and the AgentClinic NEJM runs anywhere? (Plan: no.)
- Policy folder names: `gpt5mini`, `qwen3-80b`, `deepseek-v3.2` as proposed, or the code's `gpt5mini/qwen/deepseek`.
