#!/usr/bin/env bash
set -uo pipefail

: "${HAND_DATASET_ROOT:?HAND_DATASET_ROOT is required}"
: "${DATASET_ID:?DATASET_ID is required}"
PROPOSAL_VARIANT="${PROPOSAL_VARIANT:-eos-2.0}"
HAND_LANDMARK_BACKEND="${HAND_LANDMARK_BACKEND:-rtmpose_onnx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
AUTOLABEL_CONFIG="${AUTOLABEL_CONFIG:-configs/autolabel.yaml}"
REVIEW_CONFIG="${REVIEW_CONFIG:-configs/review.yaml}"
DATASETS_CONFIG="${DATASETS_CONFIG:-configs/datasets.yaml}"
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
SOURCE_DIRS=()
while IFS= read -r -d '' source_dir; do
    if [[ -d "$source_dir/images" ]]; then
        SOURCE_DIRS+=("$source_dir")
    fi
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
if (( ${#SOURCE_DIRS[@]} == 0 )); then
    echo "ERROR: No Train capture sources with images/ found under $BASE_DIR" >&2
    exit 2
fi

DISCOVERED=${#SOURCE_DIRS[@]}
SUPPORTED_SOURCE_DIRS=()
SKIPPED_SOURCE_IDS=()
PREFLIGHT_FAILED=0
for source_dir in "${SOURCE_DIRS[@]}"; do
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    CHECK_OUTPUT="$(
        "$PYTHON_BIN" -B "$REPO_DIR/scripts/hlmf.py" \
            --autolabel-config "$AUTOLABEL_CONFIG" \
            --review-config "$REVIEW_CONFIG" \
            --datasets-config "$DATASETS_CONFIG" \
            check-palm-distance --capture-source-id "$CAPTURE_SOURCE_ID" 2>&1
    )"
    CHECK_STATUS=$?
    if (( CHECK_STATUS == 0 )); then
        SUPPORTED_SOURCE_DIRS+=("$source_dir")
    elif (( CHECK_STATUS == 3 )); then
        SKIPPED_SOURCE_IDS+=("$CAPTURE_SOURCE_ID")
        echo "SKIPPED_UNSUPPORTED_DISTANCE: $CAPTURE_SOURCE_ID"
    else
        echo "ERROR: Palm distance preflight failed for $CAPTURE_SOURCE_ID" >&2
        echo "$CHECK_OUTPUT" >&2
        PREFLIGHT_FAILED=1
    fi
done
if (( PREFLIGHT_FAILED > 0 )); then
    echo "ERROR: Train batch aborted before annotation because distance preflight failed" >&2
    exit 2
fi
if (( ${#SUPPORTED_SOURCE_DIRS[@]} == 0 )); then
    printf 'ERROR: Train batch has no Palm-compatible sources: discovered=%d skipped=%d\n' \
        "$DISCOVERED" "${#SKIPPED_SOURCE_IDS[@]}" >&2
    exit 2
fi
mkdir -p "$LOG_DIR"

TOTAL=0
SUCCESS=0
FAILED=0
printf 'Train batch: dataset=%s variant=%s backend=%s discovered=%d supported=%d skipped=%d\n' \
    "$DATASET_ID" "$PROPOSAL_VARIANT" "$HAND_LANDMARK_BACKEND" \
    "$DISCOVERED" "${#SUPPORTED_SOURCE_DIRS[@]}" "${#SKIPPED_SOURCE_IDS[@]}"

for source_dir in "${SUPPORTED_SOURCE_DIRS[@]}"; do
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    TOTAL=$((TOTAL + 1))
    LOG_FILE="$LOG_DIR/${CAPTURE_SOURCE_ID}.log"
    : > "$LOG_FILE"
    printf '[%d/%d] %s\n' "$TOTAL" "${#SUPPORTED_SOURCE_DIRS[@]}" "$CAPTURE_SOURCE_ID"

    if make -C "$REPO_DIR" train-autolabel         PYTHON="$PYTHON_BIN"         HAND_DATASET_ROOT="$HAND_DATASET_ROOT"         DATASET_SCOPE=pretrain         DATASET_ID="$DATASET_ID"         CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID"         PROPOSAL_VARIANT="$PROPOSAL_VARIANT"         HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND"         >> "$LOG_FILE" 2>&1; then
        echo "  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  train-autolabel FAILED: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

printf 'Train batch complete: discovered=%d supported=%d success=%d failed=%d skipped=%d logs=%s\n' \
    "$DISCOVERED" "${#SUPPORTED_SOURCE_DIRS[@]}" "$SUCCESS" "$FAILED" \
    "${#SKIPPED_SOURCE_IDS[@]}" "$LOG_DIR"
if (( ${#SKIPPED_SOURCE_IDS[@]} > 0 )); then
    printf 'Skipped source IDs:'
    printf ' %s' "${SKIPPED_SOURCE_IDS[@]}"
    printf '\n'
fi
if (( FAILED > 0 )); then
    exit 1
fi
