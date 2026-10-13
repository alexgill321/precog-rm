# Generating the corpora from scratch

`REPLICATION.md` starts from the released embedding corpora. This document
covers the layer before that: generating the rollouts, screening tasks, and
embedding the step texts into the `table1_data` layout. Everything here calls
paid APIs; approximate costs are given per cell. The released data lets you
skip all of it.

## Credentials and serving

| Policy | Served via | Model id | Settings |
|---|---|---|---|
| DeepSeek-V3.2 | Vertex AI MaaS (OpenAI-compatible), GCP ADC | `deepseek-ai/deepseek-v3.2-maas` (location `global`) | T=1.0 |
| Qwen3-Next-80B-Thinking | Vertex AI MaaS, GCP ADC | `qwen/qwen3-next-80b-a3b-thinking-maas` (`global`) | T=0.6, top_p=0.95; reasoning kept out of the dialogue |
| GPT-5-mini | OpenAI API | `gpt-5-mini-2025-08-07` | reasoning_effort=minimal; tau2 via chat completions, BFCL via the Responses API (`-FC` handler) |
| user simulator / NL judge (tau2) | OpenAI (LiteLLM) | `gpt-4o-mini` | |
| patient / measurement / moderator (AgentClinic) | OpenRouter (`openai/gpt-4o-mini`) or OpenAI | `gpt-4o-mini` | |
| embeddings | OpenAI | `text-embedding-3-large`, d=3072 | texts truncated to 8,000 chars; empty fields embed `<empty>` |

Environment: `OPENAI_API_KEY`; `OPENROUTER_API_KEY` (AgentClinic support roles);
`gcloud auth application-default login` plus `GCP_PROJECT_ID` for the two Vertex
policies. Never put keys in files inside the data root (`upload_hf.py` refuses
to upload credential-like content).

Pinned third-party code: tau2-bench `sierra-research/tau2-bench` @ `c5b2d22`
(install it or set `TAU2_BENCH_SRC=<checkout>/src`); `bfcl-eval==2026.3.23`
(gorilla `f7cf735`; needs its own venv because it pins numpy 1.26, see
`generation/bfcl/requirements-bfcl.txt`); AgentClinic (MIT) is vendored with
our harness changes under `generation/agentclinic/`.

## Task screening (all benchmarks)

Each task was first run 10 times with Qwen3-Next-80B-Thinking; tasks where all
10 rollouts passed or all failed were dropped. Tau2: 50 airline + 114 retail +
114 telecom official tasks -> 200 kept (39/85/76), ids in
`rollouts/meta/tau2_task_ids.json`. BFCLv3: 400 of the 800 multi-turn tasks
trialed -> 200 kept (47 base / 41 long_context / 60 miss_func / 52 miss_param),
`generation/bfcl/presets/mixed_200_tasks.json` (`400_across_cats.json` is the
trial set). AgentClinic is not pre-screened: all 214 MedQA-Extended scenarios
are rolled out and degenerate scenarios are filtered per policy afterwards.
The screen itself is the same runner with `--Z 10`.

## Tau2-Bench

```bash
bash generation/run_tau2.sh deepseek-v3.2            # -> $PRECOG_DATA/table1_data/tau2_deepseek
bash generation/run_tau2.sh qwen3-next-80b-thinking  # -> tau2_qwen
bash generation/run_tau2.sh gpt-5-mini-2025-08-07    # -> tau2_gpt5mini
```

Phases: seed the staging dir from any existing rollouts (resume), run
`generation/tau2/runner.py` (200 tasks x 50 rollouts, temperature 1.0, up to
100 steps, user simulator gpt-4o-mini, seed 42), place the flat output into
`<domain>/<idx>/rollout_NNN.json` (telecom ids via
`rollouts/meta/tau2_telecom_id_mapping.json`), then embed with
`generation/tau2/embed_tau2.py`. A step is one assistant turn: action = message
content + serialized tool calls, observation = the next tool result or user
reply (Appendix D). ~10k rollouts per policy; roughly $150 (DeepSeek), $100
(Qwen), $250 (GPT-5-mini) plus ~$40 of gpt-4o-mini per cell, and ~$15 to embed.

The released `tau2_qwen` cell was assembled from several earlier Qwen runs on
the same 200 tasks (hence >50 rollouts for some tasks; loaders cap at 50) and
carries a `manifest.json` with per-rollout provenance. Regenerating it with the
launcher gives a fresh 50-per-task cell.

## BFCLv3

```bash
python -m venv .venv_bfcl && . .venv_bfcl/bin/activate && pip install -r generation/bfcl/requirements-bfcl.txt
python -m generation.bfcl.run --ids-file generation/bfcl/presets/mixed_200_tasks.json \
    --model gpt-5-mini-2025-08-07-FC --rollouts 50 --max-concurrency 30 --run-name gpt5mini
python -m generation.bfcl.run ... --model qwen3-next-80b-thinking-vertex --run-name qwen
python -m generation.bfcl.run ... --model deepseek-v3.2-vertex --run-name deepseek
```

`generation/bfcl/run.py` wraps upstream `bfcl generate` / `bfcl evaluate` with
one `BFCL_PROJECT_ROOT` per rollout (`<run>/<category>/<task>/rollout_NN/{result,score}`),
temperature 0.8. `vertex_bfcl_handler.py` registers the two Vertex models with
upstream's handler classes. Pass provider keys through the environment (the
wrapper copies `--dotenv` into each rollout root; do not use that with the
release data root).

Embedding, two granularities (Appendix D): Qwen and DeepSeek cells split each
tool call into its own step (`embed_bfcl_percall.py`), the GPT-5-mini cell
bundles the calls of an assistant step (`embed_bfcl_bundled.py`). Then repack
into the cell layout:

```bash
python -m generation.bfcl.embed_bfcl_percall --source qwen        # -> $PRECOG_DATA/bfcl_embeddings/qwen
python -m generation.bfcl.embed_bfcl_percall --source dsv32
python -m generation.bfcl.embed_bfcl_bundled --source gpt5mini
python -m generation.bfcl.repack_bfcl_table1 --src $PRECOG_DATA/bfcl_embeddings/qwen --dst $PRECOG_DATA/table1_data/bfclv3_qwen
```

## AgentClinic (MedQA-Extended)

```bash
cd generation/agentclinic
bash run_gpt5mini_50.sh                 # OPENAI_API_KEY + OPENROUTER_API_KEY
bash run_qwen_50.sh                     # GCP ADC + OPENROUTER_API_KEY
bash run_expand_to50.sh                 # DeepSeek: GCP ADC + OPENROUTER_API_KEY
```

Outputs `scenario_<sid>_rollout_<rid>.json` per rollout (full history, tokens,
cost, correctness); move each run to `$PRECOG_DATA/rollouts/raw/agentclinic/<model>/`.
The harness is upstream AgentClinic with Vertex/OpenRouter routing,
case-insensitive `DIAGNOSIS READY` / `REQUEST TEST` matching, and per-turn
reasoning capture for the thinking policy.

Ablations of Appendix I: `run_notools_*.sh` (doctor prompt without
`REQUEST TEST`, 25 rollouts per mixed scenario; labels via
`scripts/notools_analysis.py`), `resample_diagnosis.py` (final-diagnosis
resampling; always forces a diagnosis on the resampled turn),
`expand_terra_mixed.py` (GPT-5.6-Terra reference policy).

Embedding and layout:

```bash
python -m generation.agentclinic.embed_agentclinic --variant v2 --source qwen_medqa --apply   # canonical extracted action
python -m generation.agentclinic.repack_agentclinic_table1 --tag qwen_medqa_v2             # -> table1_data/agentclinic_qwen
python -m generation.agentclinic.make_agentclinic_nodiag --source qwen_medqa              # leakage ablation cell
```

The action of a doctor turn is reduced to the canonical act (question text,
`REQUEST TEST: <test>`, or `DIAGNOSIS READY: <dx>`); the observation is the
patient's answer or the measurement result. The terminal diagnosis step is
dropped by every downstream loader (`DROP_LAST_STEP`), not at embedding time.

## From corpora to results

Continue with `REPLICATION.md`: `scripts/build_slim_caches.py`,
`scripts/pool_traj_worker.py --convert`, then the experiments.
