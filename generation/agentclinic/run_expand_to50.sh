#!/usr/bin/env bash
# Expand the DeepSeek MedQA_Ext + NEJM_Ext runs from 10 to 50 rollouts/scenario.
# Resumes into the existing output dirs (completed rollouts are skipped), so it
# only generates the missing rollouts. Safe to re-run if interrupted.
#
# Usage:
#   bash run_expand_to50.sh                 # foreground
#   nohup bash run_expand_to50.sh > outputs/expand_to50.log 2>&1 &   # detached
#
# Requires: OPENROUTER_API_KEY exported (routes gpt-4o-mini via OpenRouter) and
# GCP application-default credentials (for vertex-deepseek-v3.2).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set (needed for gpt-4o-mini via OpenRouter)." >&2
  exit 1
fi

# Silence the harmless, repeated GCP ADC "no quota project" UserWarning that
# fires on every Vertex token refresh (floods the terminal at high concurrency).
# Targeted by message + module so genuine warnings/errors still surface.
export PYTHONWARNINGS="ignore:Your application has authenticated using end user credentials:UserWarning:google.auth._default"

PY="${PYTHON:-python}"
COMMON=(--doctor_llm vertex-deepseek-v3.2
        --patient_llm gpt-4o-mini
        --measurement_llm gpt-4o-mini
        --moderator_llm gpt-4o-mini
        --num_rollouts 50
        --concurrency 30)

echo "=== [1/2] MedQA_Ext -> 50 rollouts/scenario ==="
"$PY" agentclinic.py --agent_dataset MedQA_Ext "${COMMON[@]}" \
  --output_dir outputs/medqa-ext-deepseek-10rollouts

echo "=== [2/2] NEJM_Ext -> 50 rollouts/scenario ==="
"$PY" agentclinic.py --agent_dataset NEJM_Ext "${COMMON[@]}" \
  --output_dir outputs/nejm-ext-deepseek-10rollouts

echo "=== done: both datasets at 50 rollouts/scenario ==="
