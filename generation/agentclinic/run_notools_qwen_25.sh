#!/usr/bin/env bash
# No-tools ablation, Qwen3-80b-thinking doctor (Vertex MaaS), 25 rollouts/task, on
# the 198 non-degenerate tasks (with-tools baseline SR > 0; SR=0 tasks excluded).
# Support roles (patient/measurement/moderator) = gpt-4o-mini via OpenRouter.
# Seeds from the existing 10-rollout no-tools dir so scenarios 0-19 reuse their
# first 10. Resumable. Doctor reasoning is saved per turn, never enters dialogue.
#
# Usage:  bash run_notools_qwen_25.sh
# Requires: OPENROUTER_API_KEY (support roles) + GCP application-default creds (Vertex doctor).
# NOTE: qwen throttles harder than deepseek -> concurrency 10. Run AFTER the
# deepseek run finishes (shared Vertex quota), not simultaneously.
set -euo pipefail
cd "$(dirname "$0")"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (gpt-4o-mini support roles)}"
export PYTHONWARNINGS="ignore:Your application has authenticated using end user credentials:UserWarning:google.auth._default"

SRC=outputs/medqa-ext-qwen-notools-10rollouts
DST=outputs/medqa-ext-qwen-notools-25rollouts
mkdir -p "$DST"
if [[ -d "$SRC" ]]; then
  cp -n "$SRC"/*.json "$DST"/ 2>/dev/null || true
  echo "seeded $DST from $SRC ($(ls "$SRC"/*.json 2>/dev/null | wc -l | tr -d ' ') files)"
fi

.venv/bin/python run_notools.py \
  --doctor qwen \
  --n-rollouts 25 \
  --task-ids "$(cat notools_qwen_nondegenerate.txt)" \
  --concurrency 20 \
  --out-dir "$DST"

echo "=== done: qwen no-tools, 198 non-degenerate tasks x 25 rollouts -> $DST ==="
