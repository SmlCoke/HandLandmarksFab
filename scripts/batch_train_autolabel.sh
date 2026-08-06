#!/usr/bin/env bash
set -uo pipefail

: "${HAND_DATASET_ROOT:?HAND_DATASET_ROOT is required}"
: "${DATASET_ID:?DATASET_ID is required}"
PROPOSAL_VARIANT="${PROPOSAL_VARIANT:-eos-1.0}"
HAND_LANDMARK_BACKEND="${HAND_LANDMARK_BACKEND:-rtmpose_onnx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$HAND_DATASET_ROOT/Logs/train_autolabel}"
BASE_DIR="$HAND_DATASET_ROOT/PretrainSource/$DATASET_ID"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Pretrain dataset directory does not exist: $BASE_DIR" >&2
    exit 2
fi
if [[ ! -f "$REPO_DIR/Makefile" ]]; then
    echo "ERROR: Repository Makefile does not exist: $REPO_DIR/Makefile" >&2
    exit 2
fi
mapfile -d '' SOURCE_FILES < <(
    find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type f -name source.json -print0 | sort -z
)
if (( ${#SOURCE_FILES[@]} == 0 )); then
    echo "ERROR: No registered Train capture sources found under $BASE_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR"

TOTAL=0
SUCCESS=0
FAILED=0
printf 'Train batch: dataset=%s variant=%s backend=%s sources=%d
'     "$DATASET_ID" "$PROPOSAL_VARIANT" "$HAND_LANDMARK_BACKEND" "${#SOURCE_FILES[@]}"

for source_file in "${SOURCE_FILES[@]}"; do
    source_dir="$(dirname "$source_file")"
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    TOTAL=$((TOTAL + 1))
    LOG_FILE="$LOG_DIR/${CAPTURE_SOURCE_ID}.log"
    : > "$LOG_FILE"
    printf '[%d/%d] %s
' "$TOTAL" "${#SOURCE_FILES[@]}" "$CAPTURE_SOURCE_ID"

    if make -C "$REPO_DIR" train-autolabel         PYTHON="$PYTHON_BIN"         HAND_DATASET_ROOT="$HAND_DATASET_ROOT"         DATASET_SCOPE=pretrain         DATASET_ID="$DATASET_ID"         CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID"         PROPOSAL_VARIANT="$PROPOSAL_VARIANT"         HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND"         >> "$LOG_FILE" 2>&1; then
        echo "  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  train-autolabel FAILED: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

printf 'Train batch complete: total=%d success=%d failed=%d logs=%s
'     "$TOTAL" "$SUCCESS" "$FAILED" "$LOG_DIR"
if (( FAILED > 0 )); then
    exit 1
fi
