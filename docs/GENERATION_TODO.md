# Generation layer: what is still missing from this repo (audit, 2026-10-12)

The repo reproduces everything from the embedding corpora onward. The layer
that produces those corpora is NOT yet shipped. Source files live in the
private `flow_eval` tree; all paths below are relative to it. Port them into
`generation/` with `_paths`-style roots and write `GENERATION.md`.

## Serving / credentials (document in GENERATION.md)
- DeepSeek-V3.2 and Qwen3-Next-80B-Thinking: Vertex AI MaaS, OpenAI-compatible
  endpoint, ADC auth (`GCP_PROJECT_ID` or gcloud default). Model ids
  `deepseek-ai/deepseek-v3.2-maas`, `qwen/qwen3-next-80b-a3b-thinking-maas`
  (location `global`); generation defaults DeepSeek T=1.0, Qwen T=0.6 top_p=0.95.
- GPT-5-mini: `gpt-5-mini-2025-08-07`, OpenAI API, reasoning_effort=minimal
  (tau2 via chat-completions; BFCL via Responses API, stock `-FC` handler).
- gpt-4o-mini for tau2 user simulator + NL judge (OpenAI/LiteLLM) and for
  AgentClinic patient/measurement/moderator (OpenRouter when `OPENROUTER_API_KEY` set).
- Embeddings: `text-embedding-3-large`, 3072-d, texts truncated to 8,000 chars.

## tau2 (copy `src/tau2/` -> `generation/tau2/`)
- `runner.py`, `tdp_agent.py`, `data_types.py`, `config.py`, `openai_chat_sampler.py`,
  `vertex_chat_sampler.py`, `__main__.py`. Both `runner.py` (lines ~130, ~310) and
  `vertex_chat_sampler.py` (line ~29) import from `benchmark_refs/TRAJECT-Bench/utils/model_providers.py`
  (a locally modified third-party file). Extract into `generation/tau2/vertex_maas.py`:
  `vertex_maas_models` (the two models), `vertex_maas_generation_defaults`,
  `_get_vertex_access_token`, `_get_vertex_project_id`, `_build_vertex_base_url`,
  `retry_with_backoff`, `CostTracker` (pricing table), and repoint the imports.
- Launchers: `src/experiments/run_dsv32_200.sh`, `run_gpt5mini_200.sh` -> one
  `run_tau2_200.sh --model ...` (phases: seed staging, `python -m generation.tau2.runner
  --domain all --Z 50 --N 1 --temperature 1.0 --max-steps 100 --max-errors 10 --task-split base
  --task-file meta/tau2_task_ids.json --seed 42`, reorganize into `{domain}/{idx}/rollout_NNN.json`
  using `meta/tau2_telecom_id_mapping.json`). Qwen cell was consolidated from several
  earlier runs by `src/tau2/consolidate_baselines.py` (document; the consolidated set is shipped).
- Needs tau2-bench at commit c5b2d22 (`benchmark_refs/tau2-bench`, sierra-research) on PYTHONPATH.
- Embedding: `src/prm/embed_dsv32.py` -> `generation/tau2/embed_tau2.py` (already takes `--data-dir`).
- Screening: 10-rollout qwen screen over the official task sets kept 200 mixed tasks (39/85/76).

## BFCLv3 (copy `src/bfcl_wrap/` -> `generation/bfcl/`)
- `run.py`, `bfcl_patched.py`, `vertex_bfcl_handler.py` (self-contained Vertex auth),
  `cluster_dag.py` (parsers used by the embedders). Pin `bfcl-eval==2026.3.23`
  (`exp/bfcl/requirements-bfcl.txt`; needs its own venv, numpy 1.26 pin), gorilla commit f7cf735.
- Presets: `exp/bfcl/presets/mixed_200_tasks.json` (already in `meta/`), `400_across_cats.json`.
- Command: `python -m generation.bfcl.run --ids-file meta/bfclv3_mixed_200_tasks.json
  --model {gpt-5-mini-2025-08-07-FC | qwen3-next-80b-thinking-vertex | deepseek-v3.2-vertex}
  --rollouts 50 --max-concurrency 30 --run-name <run>`; temperature 0.8.
- Embedding: `src/experiments/run_bfcl_embed_v2.py` (per-call steps; qwen + deepseek cells)
  and `run_bfcl_embed_v3.py` (bundled steps; gpt5mini cell) -> `generation/bfcl/embed_bfcl.py --granularity {call,turn}`;
  then `scripts/repack_bfcl_table1.py` -> table1_data layout.
- Screening: 400 tasks trialed with the qwen 10-rollout screen; 200 mixed kept.

## AgentClinic (copy `benchmark_refs/AgentClinic/` -> `generation/agentclinic/`)
- Modified harness `agentclinic.py` (Vertex/OpenRouter/terra routing, case-insensitive
  DIAGNOSIS READY / REQUEST TEST, per-turn reasoning capture), upstream MIT LICENSE,
  scenario files `agentclinic_medqa_extended.jsonl` (+ nejm), `requirements.txt`.
- Run scripts: `run_gpt5mini_50.sh`, `run_qwen_50.sh`, `run_expand_to50.sh` (deepseek),
  `run_notools.py` (+ `run_notools_*.sh`), `resample_diagnosis.py` (use the forced-prompt
  fix; see memory note), `expand_terra_mixed.py`. Scripts source `../../exp/bfcl/.env.*`:
  replace with plain env-var checks.
- Embedding: `src/prm/embed_agentclinic.py` (`--variant v2` = canonical extracted action;
  `_load_api_key` reads `exp/bfcl/.env.openai`: drop that fallback), then
  `scripts/repack_agentclinic_table1.py --tag <model>_medqa_v2`, `scripts/make_agentclinic_nodiag.py`.

## GRPO extras
- `src/tau2_train/eval_checkpoint.py` (base-model 20.8% eval) is missing from `src/tau2_train/`; copy it.

## Checks before shipping
- grep the copied files for `exp/`, `logs/`, `.env`, `/Users/`, `TRAJECT-Bench`; run the
  secret scan; smoke-test `--help`/dry-run of each CLI.
