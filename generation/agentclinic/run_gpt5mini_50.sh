#!/usr/bin/env bash
# AgentClinic MedQA_Ext + NEJM_Ext with the gpt-5-mini doctor
# (gpt-5-mini-2025-08-07 via api.openai.com, reasoning_effort=minimal),
# 50 rollouts/scenario. Support roles (patient/measurement/moderator) stay
# gpt-4o-mini via OpenRouter. Resumable: completed rollouts are skipped.
#
# Usage:
#   bash run_gpt5mini_50.sh             # foreground
#
# Requires: OPENAI_API_KEY (doctor) and OPENROUTER_API_KEY (support roles) —
# sourced from the repo's gitignored env files if not already exported.
# Does NOT touch Vertex, so it can run alongside the deepseek/qwen runs.
set -euo pipefail
cd "$(dirname "$0")"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY (doctor model, api.openai.com)}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (gpt-4o-mini support roles)}"

PY="${PYTHON:-python}"
COMMON=(--doctor_llm gpt-5-mini
        --patient_llm gpt-4o-mini
        --measurement_llm gpt-4o-mini
        --moderator_llm gpt-4o-mini
        --num_rollouts 50
        --concurrency 30)

echo "=== [1/2] MedQA_Ext x50 (gpt-5-mini doctor) ==="
"$PY" agentclinic.py --agent_dataset MedQA_Ext "${COMMON[@]}" \
  --output_dir outputs/medqa-ext-gpt5mini-50rollouts

echo "=== [2/2] NEJM_Ext x50 (gpt-5-mini doctor) ==="
"$PY" agentclinic.py --agent_dataset NEJM_Ext "${COMMON[@]}" \
  --output_dir outputs/nejm-ext-gpt5mini-50rollouts

echo "=== done: gpt-5-mini at 50 rollouts/scenario on both datasets ==="
