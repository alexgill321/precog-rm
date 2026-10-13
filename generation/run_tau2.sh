#!/usr/bin/env bash
# Generate the 200-task x 50-rollout tau2 cell for one policy, in the released layout.
#
#   bash generation/run_tau2.sh deepseek-v3.2            # Vertex MaaS (GCP ADC)
#   bash generation/run_tau2.sh qwen3-next-80b-thinking  # Vertex MaaS (GCP ADC)
#   bash generation/run_tau2.sh gpt-5-mini-2025-08-07    # OpenAI (OPENAI_API_KEY), reasoning_effort=minimal
#
# Requires: tau2-bench (sierra-research/tau2-bench @ c5b2d22) importable as `tau2`
# or TAU2_BENCH_SRC pointing at its src/; OPENAI_API_KEY for the gpt-4o-mini user
# simulator and NL judge (LiteLLM); $PRECOG_DATA/meta/tau2_task_ids.json.
# Resumable: existing rollouts in the cell are seeded into the staging dir first.
set -euo pipefail
MODEL="${1:?model id}"
CELL_NAME="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export PRECOG_DATA="${PRECOG_DATA:-$REPO/data}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
case "$MODEL" in
  deepseek*) CELL_NAME="${CELL_NAME:-tau2_deepseek}"; EXTRA=() ;;
  qwen*)     CELL_NAME="${CELL_NAME:-tau2_qwen}";     EXTRA=() ;;
  gpt-5*)    CELL_NAME="${CELL_NAME:-tau2_gpt5mini}"
             : "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
             EXTRA=(--api-base https://api.openai.com/v1 --api-key "$OPENAI_API_KEY") ;;
  *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac
CELL="$PRECOG_DATA/table1_data/$CELL_NAME"
STAGING="$PRECOG_DATA/rollouts/tau2_staging/$CELL_NAME"
mkdir -p "$CELL" "$STAGING"

echo "=== Phase 0: seed staging from existing cell (resume) ==="
$PY -m generation.tau2.reorganize_staging --staging "$STAGING" --cell "$CELL" --seed

echo "=== Phase 1: rollouts (200 tasks x 50, temperature 1.0, max 100 steps) ==="
$PY -m generation.tau2.runner \
  --domain all --model "$MODEL" "${EXTRA[@]}" \
  --user-model gpt-4o-mini \
  --Z 50 --N 1 --temperature 1.0 --max-steps 100 --max-errors 10 \
  --task-split base --task-file "$PRECOG_DATA/meta/tau2_task_ids.json" \
  --selection-strategy random --seed 42 --max-concurrency "${CONCURRENCY:-20}" \
  --output-dir "$STAGING"

echo "=== Phase 2: place into the cell layout ==="
$PY -m generation.tau2.reorganize_staging --staging "$STAGING" --cell "$CELL"

echo "=== Phase 3: embed (OPENAI_API_KEY) ==="
$PY -m generation.tau2.embed_tau2 --data-dir "$CELL" --apply
echo "done: $CELL"
