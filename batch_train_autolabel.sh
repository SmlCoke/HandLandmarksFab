#!/bin/bash
# ============================================================================
# Batch train-autolabel for all sources in PretrainSource.
#
# Runs Palm Detector (Eos) + RTMPose-m hand landmarker on every train
# source.  No ROI / original-image visualisation is produced.
#
# Usage (on server):
#   bash batch_train_autolabel.sh
#
# Logs are written to /root/autodl-tmp/train_autolabel_logs/<source>.log
#
# NOTE: This script will SHUT DOWN the server when finished.
# ============================================================================

set -euo pipefail

HAND_DATASET_ROOT="/root/autodl-tmp/DatesetFab"
DATASET_ID="FullEnhance0801"
PROPOSAL_VARIANT="eos-1.0"
HAND_LANDMARK_BACKEND="rtmpose_onnx"
PYTHON_BIN="/root/miniconda3/envs/anfab-rtmpose/bin/python"
REPO_DIR="/root/HandLandmarksFab"
LOG_DIR="/root/autodl-tmp/train_autolabel_logs"

mkdir -p "$LOG_DIR"

BASE_DIR="$HAND_DATASET_ROOT/PretrainSource/$DATASET_ID"

TOTAL=0
SUCCESS=0
FAILED=0

echo "========================================"
echo "Train Autolabel Batch Start: $(date)"
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

    # ---- train-autolabel -------------------------------------------------
    echo "  train-autolabel..."
    if make -C "$REPO_DIR" train-autolabel \
        PYTHON="$PYTHON_BIN" \
        HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
        DATASET_SCOPE=pretrain \
        DATASET_ID="$DATASET_ID" \
        CAPTURE_SOURCE_ID="$CAPTURE_SOURCE_ID" \
        PROPOSAL_VARIANT="$PROPOSAL_VARIANT" \
        HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND" \
        >> "$LOG_FILE" 2>&1; then
        echo "  train-autolabel  OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  train-autolabel  FAILED (see $LOG_FILE)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Train Autolabel Batch Complete: $(date)"
echo "Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "========================================"
echo ""
echo "Shutting down server in 30 seconds..."
sleep 30
shutdown -h now
