#!/bin/bash
# ============================================================================
# Batch eval-autolabel + hand-cvat-export for all sources in EValSource.
#
# Runs Palm Detector (Eos) + RTMPose-m hand landmarker on every val/test
# source, then exports CVAT XML for human review.  No ROI / original-image
# visualisation is produced.
#
# Usage (on server):
#   bash batch_eval_autolabel.sh
#
# Logs are written to /root/autodl-tmp/eval_autolabel_logs/<source>.log
# ============================================================================

set -euo pipefail

HAND_DATASET_ROOT="/root/autodl-tmp/DatesetFab"
DATASET_ID="FullEnhanceVal0801"
PROPOSAL_VARIANT="eos-1.0"
HAND_LANDMARK_BACKEND="rtmpose_onnx"
PYTHON_BIN="/root/miniconda3/envs/anfab-rtmpose/bin/python"
REPO_DIR="/root/HandLandmarksFab"
LOG_DIR="/root/autodl-tmp/eval_autolabel_logs"

mkdir -p "$LOG_DIR"

BASE_DIR="$HAND_DATASET_ROOT/EValSource/$DATASET_ID"

TOTAL=0
SUCCESS=0
FAILED=0

echo "========================================"
echo "Eval Autolabel Batch Start: $(date)"
echo "Dataset: $DATASET_ID"
echo "Backend: $HAND_LANDMARK_BACKEND"
echo "Sources: $(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "========================================"

for source_dir in "$BASE_DIR"/*/; do
    CAPTURE_SOURCE_ID=$(basename "$source_dir")
    TOTAL=$((TOTAL + 1))
    LOG_FILE="$LOG_DIR/${CAPTURE_SOURCE_ID}.log"

    echo ""
    echo "[$TOTAL] Processing: $CAPTURE_SOURCE_ID"
    echo "      Log: $LOG_FILE"

    # ---- Step 1: eval-autolabel ------------------------------------------
    echo "  [1/2] eval-autolabel..."
    if make -C "$REPO_DIR" eval-autolabel \
        PYTHON="$PYTHON_BIN" \
        HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
        DATASET_SCOPE=eval \
        DATASET_ID="$DATASET_ID" \
        CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID" \
        PROPOSAL_VARIANT="$PROPOSAL_VARIANT" \
        HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND" \
        >> "$LOG_FILE" 2>&1; then
        echo "  [1/2] eval-autolabel  OK"
    else
        echo "  [1/2] eval-autolabel  FAILED (see $LOG_FILE)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ---- Step 2: hand-cvat-export ---------------------------------------
    echo "  [2/2] hand-cvat-export..."
    if make -C "$REPO_DIR" hand-cvat-export \
        PYTHON="$PYTHON_BIN" \
        HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
        DATASET_SCOPE=eval \
        DATASET_ID="$DATASET_ID" \
        CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID" \
        PROPOSAL_VARIANT="$PROPOSAL_VARIANT" \
        >> "$LOG_FILE" 2>&1; then
        echo "  [2/2] hand-cvat-export  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  [2/2] hand-cvat-export  FAILED (see $LOG_FILE)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Eval Autolabel Batch Complete: $(date)"
echo "Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "========================================"
