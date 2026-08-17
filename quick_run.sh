cd ~/HandLandmarksFab
conda activate anfab

# Autolabel by Eos-2.1 + HaMeR + hcf: v1-mobilenet-v3-large
make batch-train-autolabel DATASET_ID=FullEnhance0801 PROPOSAL_VARIANT=eos_2.1-hamer-v1mv3l-gate HAND_LANDMARK_BACKEND=hamer
make batch-train-autolabel DATASET_ID=FullEnhance0803 PROPOSAL_VARIANT=eos_2.1-hamer-v1mv3l-gate HAND_LANDMARK_BACKEND=hamer
make batch-train-autolabel DATASET_ID=FullEnhance0810 PROPOSAL_VARIANT=eos_2.1-hamer-v1mv3l-gate HAND_LANDMARK_BACKEND=hamer
make batch-train-autolabel DATASET_ID=FullEnhance0817 PROPOSAL_VARIANT=eos_2.1-hamer-v1mv3l-gate HAND_LANDMARK_BACKEND=hamer

cd ~/HandLandmarkerLab/
conda activate hand-landmarker-tf29 
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_TRAIN_ROOT=/root/autodl-tmp/TrainFab/HLML-4.0
export HLML_SNAPSHOT_ID=Iris_1.2-Eos_2.1-hcf_v1mv3l-r1
export HLML_EXPERIMENT_ID=Iris_1.2-Eos_2.1-hcf_v1mv3l-r1
export HLML_RELEASE_ID=Iris_1.2-Eos_2.1-hcf_v1mv3l-r1

# check and audit
make config-check
make data-audit HLML_STAGE=geometry

make geometry
make val HLML_STAGE=geometry
