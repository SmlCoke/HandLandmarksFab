.DEFAULT_GOAL := help

PYTHON ?= python
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
DATASET_SCOPE ?= pretrain
DATASET_ID ?=
CAPTURE_SOURCE_ID ?=
PROPOSAL_VARIANT ?= eos-1.0
AUTOLABEL_CONFIG ?= configs/autolabel.yaml
REVIEW_CONFIG ?= configs/review.yaml
DATASETS_CONFIG ?= configs/datasets.yaml
ROI_VISUALIZATION ?=
ORIGINAL_VISUALIZATION ?=
HAND_LANDMARK_BACKEND ?=

NEGATIVE_DATASET_ID ?=
NEGATIVE_CANDIDATE_LABELS ?=
SELECTION_ID ?=
MINING_REQUEST ?=

CLI = $(PYTHON) -B scripts/hlmf.py --autolabel-config "$(AUTOLABEL_CONFIG)" --review-config "$(REVIEW_CONFIG)" --datasets-config "$(DATASETS_CONFIG)"
SOURCE_ARGS = --dataset-root "$(HAND_DATASET_ROOT)" --scope "$(DATASET_SCOPE)" --dataset-id "$(DATASET_ID)" --capture-source-id "$(CAPTURE_SOURCE_ID)" --proposal-variant "$(PROPOSAL_VARIANT)"
AUTOLABEL_ARGS = $(if $(strip $(ROI_VISUALIZATION)),--roi-visualization "$(ROI_VISUALIZATION)",) \
	$(if $(strip $(ORIGINAL_VISUALIZATION)),--original-visualization "$(ORIGINAL_VISUALIZATION)",) \
	$(if $(strip $(HAND_LANDMARK_BACKEND)),--hand-landmark-backend "$(HAND_LANDMARK_BACKEND)",)

.PHONY: help paths source-check train-autolabel eval-autolabel autolabel-visualize-roi \
	autolabel-visualize-original hand-cvat-export \
	hand-cvat-import source-publish negative-review negative-publish hard-review \
	hard-publish registry-check compile test

help:
	@echo HLMF 3.0 - Palm proposals to versioned Hand ROI datasets
	@echo Configs: autolabel.yaml=automatic labels, review.yaml=Hand CVAT, datasets.yaml=publication, cvat_label.json=CVAT schema
	@echo   make source-check DATASET_SCOPE=pretrain/eval DATASET_ID=... CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-1.0
	@echo   make train-autolabel ... [HAND_LANDMARK_BACKEND=mediapipe_tasks/rtmpose_onnx] [ROI_VISUALIZATION=true/false] [ORIGINAL_VISUALIZATION=true/false]
	@echo   make eval-autolabel ... [HAND_LANDMARK_BACKEND=mediapipe_tasks/rtmpose_onnx] [ROI_VISUALIZATION=true/false] [ORIGINAL_VISUALIZATION=true/false]
	@echo   make autolabel-visualize-roi ...  Render existing draft on Hand ROI images
	@echo   make autolabel-visualize-original ...  Render existing draft on original images
	@echo   make hand-cvat-export ...         Export Hand ROI CVAT XML only
	@echo   make hand-cvat-import ...         Import reviewed Hand ROI XML
	@echo   make source-publish ...           Publish reviewed Val/Test labels
	@echo   make negative-review NEGATIVE_DATASET_ID=... NEGATIVE_CANDIDATE_LABELS=/abs/candidates.jsonl
	@echo   make negative-publish NEGATIVE_DATASET_ID=...
	@echo   make hard-review SELECTION_ID=... MINING_REQUEST=/abs/request.jsonl
	@echo   make hard-publish SELECTION_ID=...
	@echo   make registry-check
	@echo No command exports, imports or edits Palm annotations; Hand ROIs are always program-generated.

paths:
	@echo HAND_DATASET_ROOT=$(HAND_DATASET_ROOT)
	@echo DATASET_SCOPE=$(DATASET_SCOPE)
	@echo DATASET_ID=$(DATASET_ID)
	@echo CAPTURE_SOURCE_ID=$(CAPTURE_SOURCE_ID)
	@echo PROPOSAL_VARIANT=$(PROPOSAL_VARIANT)

source-check:
	$(CLI) validate-source $(SOURCE_ARGS)

train-autolabel:
	$(CLI) autolabel-train $(SOURCE_ARGS) $(AUTOLABEL_ARGS)

eval-autolabel:
	$(CLI) autolabel-eval $(SOURCE_ARGS) $(AUTOLABEL_ARGS)

autolabel-visualize-roi:
	$(CLI) autolabel-visualize-roi $(SOURCE_ARGS)

autolabel-visualize-original:
	$(CLI) autolabel-visualize-original $(SOURCE_ARGS)

hand-cvat-export:
	$(CLI) export-cvat $(SOURCE_ARGS)

hand-cvat-import:
	$(CLI) import-cvat $(SOURCE_ARGS)

source-publish:
	$(CLI) publish-source $(SOURCE_ARGS)

negative-review:
	$(if $(strip $(NEGATIVE_DATASET_ID)),,$(error NEGATIVE_DATASET_ID is required))
	$(if $(strip $(NEGATIVE_CANDIDATE_LABELS)),,$(error NEGATIVE_CANDIDATE_LABELS is required))
	$(CLI) prepare-negative-review --dataset-root "$(HAND_DATASET_ROOT)" --negative-dataset-id "$(NEGATIVE_DATASET_ID)" --candidate-labels "$(NEGATIVE_CANDIDATE_LABELS)"

negative-publish:
	$(if $(strip $(NEGATIVE_DATASET_ID)),,$(error NEGATIVE_DATASET_ID is required))
	$(CLI) publish-negative-review --dataset-root "$(HAND_DATASET_ROOT)" --negative-dataset-id "$(NEGATIVE_DATASET_ID)"

hard-review:
	$(if $(strip $(SELECTION_ID)),,$(error SELECTION_ID is required))
	$(if $(strip $(MINING_REQUEST)),,$(error MINING_REQUEST is required))
	$(CLI) prepare-selection-review --dataset-root "$(HAND_DATASET_ROOT)" --selection-id "$(SELECTION_ID)" --request "$(MINING_REQUEST)"

hard-publish:
	$(if $(strip $(SELECTION_ID)),,$(error SELECTION_ID is required))
	$(CLI) publish-selection-review --dataset-root "$(HAND_DATASET_ROOT)" --selection-id "$(SELECTION_ID)"

registry-check:
	$(CLI) registry-check --dataset-root "$(HAND_DATASET_ROOT)"

compile:
	$(PYTHON) -B -c "from pathlib import Path; files=[p for root in ('hand_autolabel','scripts','tests','tools') for p in Path(root).rglob('*.py')]; [compile(p.read_bytes(), str(p), 'exec') for p in files]; print('syntax-checked', len(files), 'Python files')"

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py"
