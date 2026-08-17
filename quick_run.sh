#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

readonly HLMF_REPO=/root/HandLandmarksFab
readonly HLML_REPO=/root/HandLandmarkerLab
readonly PROPOSAL_VARIANT=eos_2.1-hamer-v1mv3l-gate
readonly HAND_LANDMARK_BACKEND=hamer
readonly RUN_ID=iris-1.2-geometry-eos2.1-hamer-hcf-v1mv3l-r1
readonly -a DATASET_IDS=(
  FullEnhance0801
  FullEnhance0803
  FullEnhance0810
  FullEnhance0817
)

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_SNAPSHOT_ID="$RUN_ID"
export HLML_EXPERIMENT_ID="$RUN_ID"
export HLML_RELEASE_ID="$RUN_ID"
export HLML_STAGE=geometry

# Parse all HLML public configs before starting the expensive annotation stage.
conda activate hand-landmarker-tf29
cd "$HLML_REPO"
make config-check

# Publish the Eos-2.1 + HaMeR + HCF v1 variant for every supported Train source.
conda activate anfab
cd "$HLMF_REPO"
for dataset_id in "${DATASET_IDS[@]}"; do
  make batch-train-autolabel \
    HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
    DATASET_ID="$dataset_id" \
    PROPOSAL_VARIANT="$PROPOSAL_VARIANT" \
    HAND_LANDMARK_BACKEND="$HAND_LANDMARK_BACKEND"
done

# The geometry target performs its own data audit before training.
conda activate hand-landmarker-tf29
cd "$HLML_REPO"
make geometry
make val
