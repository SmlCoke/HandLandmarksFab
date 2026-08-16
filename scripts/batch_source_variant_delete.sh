#!/usr/bin/env bash
set -uo pipefail

: "${HAND_DATASET_ROOT:?HAND_DATASET_ROOT is required}"
: "${DATASET_ID:?DATASET_ID is required}"
: "${CONFIRM_DELETE:?CONFIRM_DELETE is required}"
DATASET_SCOPE="${DATASET_SCOPE:-pretrain}"
PROPOSAL_VARIANT="${PROPOSAL_VARIANT:-eos-2.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ "$CONFIRM_DELETE" != "$PROPOSAL_VARIANT" ]]; then
    echo "ERROR: CONFIRM_DELETE must exactly match PROPOSAL_VARIANT" >&2
    exit 2
fi
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
    if [[ -f "$source_dir/source.json" ]]; then
        SOURCE_DIRS+=("$source_dir")
    fi
done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
if (( ${#SOURCE_DIRS[@]} == 0 )); then
    echo "ERROR: No registered capture sources found under $BASE_DIR" >&2
    exit 2
fi

TOTAL=0
SUCCESS=0
FAILED=0
printf 'Source variant deletion batch: scope=%s dataset=%s variant=%s sources=%d\n' \
    "$DATASET_SCOPE" "$DATASET_ID" "$PROPOSAL_VARIANT" "${#SOURCE_DIRS[@]}"

for source_dir in "${SOURCE_DIRS[@]}"; do
    CAPTURE_SOURCE_ID="$(basename "$source_dir")"
    TOTAL=$((TOTAL + 1))
    printf '[%d/%d] %s\n' "$TOTAL" "${#SOURCE_DIRS[@]}" "$CAPTURE_SOURCE_ID"
    if make -C "$REPO_DIR" source-variant-delete \
        PYTHON="$PYTHON_BIN" HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
        DATASET_SCOPE="$DATASET_SCOPE" DATASET_ID="$DATASET_ID" \
        CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID" PROPOSAL_VARIANT="$PROPOSAL_VARIANT" \
        CONFIRM_DELETE="$CONFIRM_DELETE"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

MANIFEST_FAILED=0
if make -s -C "$REPO_DIR" dataset-manifest-rebuild \
    PYTHON="$PYTHON_BIN" HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
    DATASET_SCOPE="$DATASET_SCOPE" DATASET_ID="$DATASET_ID" >/dev/null; then
    echo "Dataset manifest rebuilt: $BASE_DIR/dataset_manifest.json"
else
    echo "ERROR: Dataset manifest rebuild failed" >&2
    MANIFEST_FAILED=1
fi

printf 'Source variant deletion complete: total=%d success=%d failed=%d manifest_failed=%d\n' \
    "$TOTAL" "$SUCCESS" "$FAILED" "$MANIFEST_FAILED"
if (( FAILED > 0 || MANIFEST_FAILED > 0 )); then
    exit 1
fi
