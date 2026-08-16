#!/usr/bin/env bash
set -uo pipefail

: "${HAND_DATASET_ROOT:?HAND_DATASET_ROOT is required}"
: "${DATASET_ID:?DATASET_ID is required}"
DATASET_SCOPE="${DATASET_SCOPE:-pretrain}"
PROPOSAL_VARIANT="${PROPOSAL_VARIANT:-eos-2.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

case "$DATASET_SCOPE" in
    pretrain) BUCKET="PretrainSource" ;;
    eval) BUCKET="EValSource" ;;
    *)
        echo "ERROR: DATASET_SCOPE must be pretrain or eval: $DATASET_SCOPE" >&2
        exit 2
        ;;
esac
BASE_DIR="$HAND_DATASET_ROOT/$BUCKET/$DATASET_ID"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Dataset directory does not exist: $BASE_DIR" >&2
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
    echo "ERROR: No capture sources with images/ found under $BASE_DIR" >&2
    exit 2
fi

TOTAL=0
SUCCESS=0
FAILED=0
printf 'Visualization cleanup batch: scope=%s dataset=%s variant=%s sources=%d\n' \
    "$DATASET_SCOPE" "$DATASET_ID" "$PROPOSAL_VARIANT" "${#SOURCE_DIRS[@]}"

for source_dir in "${SOURCE_DIRS[@]}"; do
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    TOTAL=$((TOTAL + 1))
    printf '[%d/%d] %s\n' "$TOTAL" "${#SOURCE_DIRS[@]}" "$CAPTURE_SOURCE_ID"
    if make -C "$REPO_DIR" autolabel-visualizations-clean \
        PYTHON="$PYTHON_BIN" HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
        DATASET_SCOPE="$DATASET_SCOPE" DATASET_ID="$DATASET_ID" \
        CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID" PROPOSAL_VARIANT="$PROPOSAL_VARIANT"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

printf 'Visualization cleanup complete: total=%d success=%d failed=%d\n' \
    "$TOTAL" "$SUCCESS" "$FAILED"
if (( FAILED > 0 )); then
    exit 1
fi
