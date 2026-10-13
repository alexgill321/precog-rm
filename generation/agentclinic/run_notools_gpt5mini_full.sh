#!/usr/bin/env bash
# No-tools ablation, gpt-5-mini doctor, FULL MedQA_Ext set (214 scenarios) x 25
# rollouts. Doctor prompt has the REQUEST TEST capability removed; a scripted
# refusal is returned if the doctor asks for a test anyway (see run_notools.py).
#
# Reuses the existing 200 no-tools rollouts (scenarios 0-19, 10 each) by seeding
# the target dir, so only the ~5150 missing rollouts are generated. Resumable:
# completed files are skipped on re-run.
#
# Usage:  bash run_notools_gpt5mini_full.sh
#
# Everything goes through api.openai.com: the gpt-5-mini doctor AND the
# gpt-4o-mini support roles (patient/measurement/moderator). OpenRouter is NOT
# used — OPENROUTER_API_KEY is unset below so agentclinic's shared client routes
# gpt-4o-mini direct to OpenAI (bare "gpt-4o-mini" id).
set -euo pipefail
cd "$(dirname "$0")"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY (all roles, api.openai.com)}"
# Force direct OpenAI for the support roles even if these leak in from the shell.
unset OPENROUTER_API_KEY OPENAI_BASE_URL

SRC=outputs/medqa-ext-gpt5mini-notools-10rollouts
DST=outputs/medqa-ext-gpt5mini-notools-25rollouts
mkdir -p "$DST"

# Seed with the existing 200 rollouts (rollout_00..09 for scenarios 0-19) so they
# count toward the 25 and are not regenerated. cp -n never overwrites.
if [[ -d "$SRC" ]]; then
  n=$(ls "$SRC"/*.json 2>/dev/null | wc -l | tr -d ' ')
  cp -n "$SRC"/*.json "$DST"/ 2>/dev/null || true
  echo "seeded $DST with $n existing no-tools rollouts from $SRC"
fi

PY="${PYTHON:-python}"
"$PY" run_notools.py \
  --doctor gpt5mini \
  --n-rollouts 25 \
  --tasks 0 214 \
  --concurrency 30 \
  --out-dir "$DST"

echo "=== done: gpt-5-mini no-tools, 214 scenarios x 25 rollouts -> $DST ==="
