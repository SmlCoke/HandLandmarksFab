#!/usr/bin/env bash
set -uo pipefail

: "${HAND_DATASET_ROOT:?HAND_DATASET_ROOT is required}"
: "${DATASET_ID:?DATASET_ID is required}"
PROPOSAL_VARIANT="${PROPOSAL_VARIANT:-eos-2.0}"
HAND_LANDMARK_BACKEND="${HAND_LANDMARK_BACKEND:-rtmpose_onnx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$HAND_DATASET_ROOT/Logs/eval_autolabel}"
BASE_DIR="$HAND_DATASET_ROOT/EValSource/$DATASET_ID"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Eval dataset directory does not exist: $BASE_DIR" >&2
    exit 2
fi
if [[ ! -f "$REPO_DIR/Makefile" ]]; then
    echo "ERROR: Repository Makefile does not exist: $REPO_DIR/Makefile" >&2
    exit 2
fi
SOURCE_DIRS=()
while IFS= read -r -d '' source_dir; do
    if [[ -d "$source_dir/images" ]]; then
        SOURCE_DIRS+=("$source_dir")
    fi
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
if (( ${#SOURCE_DIRS[@]} == 0 )); then
    echo "ERROR: No Eval capture sources with images/ found under $BASE_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR"

TOTAL=0
SUCCESS=0
FAILED=0
printf 'Eval batch: dataset=%s variant=%s backend=%s sources=%d
'     "$DATASET_ID" "$PROPOSAL_VARIANT" "$HAND_LANDMARK_BACKEND" "${#SOURCE_DIRS[@]}"

for source_dir in "${SOURCE_DIRS[@]}"; do
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    TOTAL=$((TOTAL + 1))
    LOG_FILE="$LOG_DIR/${CAPTURE_SOURCE_ID}.log"
    : > "$LOG_FILE"
    printf '[%d/%d] %s
' "$TOTAL" "${#SOURCE_DIRS[@]}" "$CAPTURE_SOURCE_ID"

    if ! make -C "$REPO_DIR" eval-autolabel         PYTHON="$PYTHON_BIN"         HAND_DATASET_ROOT="$HAND_DATASET_ROOT"         DATASET_SCOPE=eval         DATASET_ID="$DATASET_ID"         CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID"         PROPOSAL_VARIANT="$PROPOSAL_VARIANT"         HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND"         >> "$LOG_FILE" 2>&1; then
        echo "  eval-autolabel FAILED: $LOG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    if make -C "$REPO_DIR" hand-cvat-export         PYTHON="$PYTHON_BIN"         HAND_DATASET_ROOT="$HAND_DATASET_ROOT"         DATASET_SCOPE=eval         DATASET_ID="$DATASET_ID"         CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID"         PROPOSAL_VARIANT="$PROPOSAL_VARIANT"         >> "$LOG_FILE" 2>&1; then
        echo "  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  hand-cvat-export FAILED: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

printf 'Eval batch complete: total=%d success=%d failed=%d logs=%s
'     "$TOTAL" "$SUCCESS" "$FAILED" "$LOG_DIR"
if (( FAILED > 0 )); then
    exit 1
fi
