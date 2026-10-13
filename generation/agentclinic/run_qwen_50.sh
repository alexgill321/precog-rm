#!/usr/bin/env bash
# AgentClinic MedQA_Ext + NEJM_Ext with the Qwen thinking doctor
# (vertex-qwen3-80b -> qwen/qwen3-next-80b-a3b-thinking-maas), 50 rollouts/scenario.
# Doctor reasoning is saved on each doctor turn in the rollout files but never
# enters the dialogue. Resumable: completed rollouts are skipped on re-run.
#
# Usage:
#   bash run_qwen_50.sh                 # foreground
#
# Requires: GCP application-default credentials (Vertex doctor) and an
# OpenRouter key (gpt-4o-mini support roles) — sourced from the repo's
# gitignored env file if not already exported.
#
# NOTE: shares Vertex quota with any running deepseek expansion
# (run_expand_to50.sh) — prefer to run after it finishes.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set (needed for gpt-4o-mini via OpenRouter)." >&2
  exit 1
fi

# Silence the harmless, repeated GCP ADC "no quota project" UserWarning.
export PYTHONWARNINGS="ignore:Your application has authenticated using end user credentials:UserWarning:google.auth._default"

PY="${PYTHON:-python}"
COMMON=(--doctor_llm vertex-qwen3-80b
        --patient_llm gpt-4o-mini
        --measurement_llm gpt-4o-mini
        --moderator_llm gpt-4o-mini
        --num_rollouts 50
        --concurrency 10)

echo "=== [1/2] MedQA_Ext x50 (qwen doctor) ==="
"$PY" agentclinic.py --agent_dataset MedQA_Ext "${COMMON[@]}" \
  --output_dir outputs/medqa-ext-qwen-50rollouts

echo "=== [2/2] NEJM_Ext x50 (qwen doctor) ==="
"$PY" agentclinic.py --agent_dataset NEJM_Ext "${COMMON[@]}" \
  --output_dir outputs/nejm-ext-qwen-50rollouts

echo "=== done: qwen at 50 rollouts/scenario on both datasets ==="
