#!/usr/bin/env bash
# No-tools ablation, DeepSeek-v3.2 doctor (Vertex MaaS), 25 rollouts/task, on the
# 209 non-degenerate tasks (with-tools baseline SR > 0; SR=0 tasks excluded since
# they can't show a drop). Support roles (patient/measurement/moderator) =
# gpt-4o-mini via OpenRouter. Seeds from the existing 10-rollout no-tools dir so
# scenarios 0-19 reuse their first 10. Resumable.
#
# Usage:  bash run_notools_deepseek_25.sh
# Requires: OPENROUTER_API_KEY (support roles) + GCP application-default creds (Vertex doctor).
# NOTE: shares Vertex project quota with the qwen run — run them SEQUENTIALLY, not at once.
set -euo pipefail
cd "$(dirname "$0")"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (gpt-4o-mini support roles)}"
export PYTHONWARNINGS="ignore:Your application has authenticated using end user credentials:UserWarning:google.auth._default"

SRC=outputs/medqa-ext-deepseek-notools-10rollouts
DST=outputs/medqa-ext-deepseek-notools-25rollouts
mkdir -p "$DST"
if [[ -d "$SRC" ]]; then
  cp -n "$SRC"/*.json "$DST"/ 2>/dev/null || true
  echo "seeded $DST from $SRC ($(ls "$SRC"/*.json 2>/dev/null | wc -l | tr -d ' ') files)"
fi

.venv/bin/python run_notools.py \
  --doctor deepseek \
  --n-rollouts 25 \
  --task-ids "$(cat notools_deepseek_nondegenerate.txt)" \
  --concurrency 20 \
  --out-dir "$DST"

echo "=== done: deepseek no-tools, 209 non-degenerate tasks x 25 rollouts -> $DST ==="
