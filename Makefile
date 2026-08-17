.DEFAULT_GOAL := help

PYTHON ?= python
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
DATASET_SCOPE ?= pretrain
DATASET_ID ?=
CAPTURE_SOURCE_ID ?=
PROPOSAL_VARIANT ?= eos-2.1
AUTOLABEL_CONFIG ?= configs/autolabel.yaml
REVIEW_CONFIG ?= configs/review.yaml
DATASETS_CONFIG ?= configs/datasets.yaml
ROI_VISUALIZATION ?=
ORIGINAL_VISUALIZATION ?=
ORIGINAL_VIDEO ?=
HAND_LANDMARK_BACKEND ?=
CONFIRM_DELETE ?=
BATCH_LOG_DIR ?=

NEGATIVE_DATASET_ID ?=
NEGATIVE_CANDIDATE_LABELS ?=
MINING_REQUEST ?=
HARD_DATASET_ID ?=

CLI = $(PYTHON) -B scripts/hlmf.py --autolabel-config "$(AUTOLABEL_CONFIG)" --review-config "$(REVIEW_CONFIG)" --datasets-config "$(DATASETS_CONFIG)"
SOURCE_ARGS = --dataset-root "$(HAND_DATASET_ROOT)" --scope "$(DATASET_SCOPE)" --dataset-id "$(DATASET_ID)" --capture-source-id "$(CAPTURE_SOURCE_ID)" --proposal-variant "$(PROPOSAL_VARIANT)"
AUTOLABEL_ARGS = $(if $(strip $(ROI_VISUALIZATION)),--roi-visualization "$(ROI_VISUALIZATION)",) \
	$(if $(strip $(ORIGINAL_VISUALIZATION)),--original-visualization "$(ORIGINAL_VISUALIZATION)",) \
	$(if $(strip $(HAND_LANDMARK_BACKEND)),--hand-landmark-backend "$(HAND_LANDMARK_BACKEND)",)

.PHONY: help paths source-check palm-distance-check train-autolabel eval-autolabel autolabel-visualize-roi \
	autolabel-visualize-original autolabel-visualizations-clean \
	batch-autolabel-visualizations-clean source-variant-delete batch-source-variant-delete \
	dataset-manifest-rebuild batch-eval-autolabel batch-train-autolabel hand-cvat-export \
	hand-cvat-import source-publish gold-autolabel negative-review negative-publish \
	hard-review hard-import hard-publish registry-check compile test

help:
	@echo HLMF 3.0 - Palm proposals to versioned Hand ROI datasets
	@echo Configs: autolabel.yaml=automatic labels, review.yaml=Hand CVAT, datasets.yaml=publication, cvat_label.json=CVAT schema
	@echo   make source-check DATASET_SCOPE=pretrain/eval/gold DATASET_ID=... CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.1
	@echo   make palm-distance-check CAPTURE_SOURCE_ID=...  Check current Palm model distance support
	@echo   make train-autolabel ... [HAND_LANDMARK_BACKEND=mediapipe_tasks/rtmpose_onnx/hamer] [ROI_VISUALIZATION=true/false] [ORIGINAL_VISUALIZATION=true/false]
	@echo   make eval-autolabel ... [HAND_LANDMARK_BACKEND=mediapipe_tasks/rtmpose_onnx/hamer] [ROI_VISUALIZATION=true/false] [ORIGINAL_VISUALIZATION=true/false]
	@echo   make gold-autolabel DATASET_SCOPE=gold ...  Auto-label a new recorded Gold Train source before CVAT review
	@echo   make autolabel-visualize-roi ...  Render existing draft on Hand ROI images
	@echo   make autolabel-visualize-original ... [ORIGINAL_VIDEO=true/false]
	@echo   make autolabel-visualizations-clean ...  Remove rebuildable visualization outputs
	@echo   make batch-autolabel-visualizations-clean DATASET_SCOPE=... DATASET_ID=... PROPOSAL_VARIANT=...
	@echo   make source-variant-delete ... CONFIRM_DELETE=exact-variant
	@echo   make batch-source-variant-delete DATASET_SCOPE=... DATASET_ID=... PROPOSAL_VARIANT=... CONFIRM_DELETE=exact-variant
	@echo   make dataset-manifest-rebuild DATASET_SCOPE=... DATASET_ID=...
	@echo   make batch-eval-autolabel DATASET_ID=... PROPOSAL_VARIANT=...
	@echo   make batch-train-autolabel DATASET_ID=... PROPOSAL_VARIANT=...
	@echo   make hand-cvat-export ...         Export Hand ROI CVAT XML only
	@echo   make hand-cvat-import ...         Import reviewed Hand ROI XML
	@echo   make source-publish ...           Publish reviewed Eval/Gold labels
	@echo   make negative-review NEGATIVE_DATASET_ID=... NEGATIVE_CANDIDATE_LABELS=/abs/candidates.jsonl
	@echo   make negative-publish NEGATIVE_DATASET_ID=...
	@echo   make hard-review HARD_DATASET_ID=... MINING_REQUEST=/abs/request.jsonl
	@echo   make hard-import HARD_DATASET_ID=...  Import review/cvat_reviewed.xml
	@echo   make hard-publish HARD_DATASET_ID=...
	@echo   make registry-check
	@echo "No command exports, imports or edits Palm annotations; Hand ROIs are always program-generated."

paths:
	@echo HAND_DATASET_ROOT=$(HAND_DATASET_ROOT)
	@echo DATASET_SCOPE=$(DATASET_SCOPE)
	@echo DATASET_ID=$(DATASET_ID)
	@echo CAPTURE_SOURCE_ID=$(CAPTURE_SOURCE_ID)
	@echo PROPOSAL_VARIANT=$(PROPOSAL_VARIANT)

source-check:
	$(CLI) validate-source $(SOURCE_ARGS)

palm-distance-check:
	$(CLI) check-palm-distance --capture-source-id "$(CAPTURE_SOURCE_ID)"

train-autolabel:
	$(CLI) autolabel-train $(SOURCE_ARGS) $(AUTOLABEL_ARGS)

eval-autolabel:
	$(CLI) autolabel-eval $(SOURCE_ARGS) $(AUTOLABEL_ARGS)

gold-autolabel:
	$(CLI) autolabel-gold $(SOURCE_ARGS) $(AUTOLABEL_ARGS)

autolabel-visualize-roi:
	$(CLI) autolabel-visualize-roi $(SOURCE_ARGS)

autolabel-visualize-original:
	$(CLI) autolabel-visualize-original $(SOURCE_ARGS) $(if $(strip $(ORIGINAL_VIDEO)),--original-video "$(ORIGINAL_VIDEO)",)

autolabel-visualizations-clean:
	$(CLI) clean-autolabel-visualizations $(SOURCE_ARGS)

batch-autolabel-visualizations-clean:
	$(if $(strip $(DATASET_ID)),,$(error DATASET_ID is required))
	HAND_DATASET_ROOT="$(HAND_DATASET_ROOT)" DATASET_SCOPE="$(DATASET_SCOPE)" DATASET_ID="$(DATASET_ID)" PROPOSAL_VARIANT="$(PROPOSAL_VARIANT)" PYTHON_BIN="$(PYTHON)" REPO_DIR="$(CURDIR)" bash scripts/batch_autolabel_visualizations_clean.sh

source-variant-delete:
	$(if $(strip $(CONFIRM_DELETE)),,$(error CONFIRM_DELETE must exactly match PROPOSAL_VARIANT))
	$(CLI) delete-source-variant $(SOURCE_ARGS) --confirm-delete "$(CONFIRM_DELETE)"

batch-source-variant-delete:
	$(if $(strip $(DATASET_ID)),,$(error DATASET_ID is required))
	$(if $(strip $(CONFIRM_DELETE)),,$(error CONFIRM_DELETE must exactly match PROPOSAL_VARIANT))
	HAND_DATASET_ROOT="$(HAND_DATASET_ROOT)" DATASET_SCOPE="$(DATASET_SCOPE)" DATASET_ID="$(DATASET_ID)" PROPOSAL_VARIANT="$(PROPOSAL_VARIANT)" CONFIRM_DELETE="$(CONFIRM_DELETE)" PYTHON_BIN="$(PYTHON)" REPO_DIR="$(CURDIR)" bash scripts/batch_source_variant_delete.sh

dataset-manifest-rebuild:
	$(if $(strip $(DATASET_ID)),,$(error DATASET_ID is required))
	$(CLI) rebuild-dataset-manifest --dataset-root "$(HAND_DATASET_ROOT)" --scope "$(DATASET_SCOPE)" --dataset-id "$(DATASET_ID)"

batch-eval-autolabel:
	$(if $(strip $(DATASET_ID)),,$(error DATASET_ID is required))
	HAND_DATASET_ROOT="$(HAND_DATASET_ROOT)" DATASET_ID="$(DATASET_ID)" PROPOSAL_VARIANT="$(PROPOSAL_VARIANT)" HAND_LANDMARK_BACKEND="$(or $(HAND_LANDMARK_BACKEND),rtmpose_onnx)" PYTHON_BIN="$(PYTHON)" REPO_DIR="$(CURDIR)" AUTOLABEL_CONFIG="$(AUTOLABEL_CONFIG)" REVIEW_CONFIG="$(REVIEW_CONFIG)" DATASETS_CONFIG="$(DATASETS_CONFIG)" $(if $(strip $(BATCH_LOG_DIR)),LOG_DIR="$(BATCH_LOG_DIR)",) bash scripts/batch_eval_autolabel.sh

batch-train-autolabel:
	$(if $(strip $(DATASET_ID)),,$(error DATASET_ID is required))
	HAND_DATASET_ROOT="$(HAND_DATASET_ROOT)" DATASET_ID="$(DATASET_ID)" PROPOSAL_VARIANT="$(PROPOSAL_VARIANT)" HAND_LANDMARK_BACKEND="$(or $(HAND_LANDMARK_BACKEND),rtmpose_onnx)" PYTHON_BIN="$(PYTHON)" REPO_DIR="$(CURDIR)" AUTOLABEL_CONFIG="$(AUTOLABEL_CONFIG)" REVIEW_CONFIG="$(REVIEW_CONFIG)" DATASETS_CONFIG="$(DATASETS_CONFIG)" $(if $(strip $(BATCH_LOG_DIR)),LOG_DIR="$(BATCH_LOG_DIR)",) bash scripts/batch_train_autolabel.sh

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
	$(if $(strip $(HARD_DATASET_ID)),,$(error HARD_DATASET_ID is required))
	$(if $(strip $(MINING_REQUEST)),,$(error MINING_REQUEST is required))
	$(CLI) prepare-hard-review --dataset-root "$(HAND_DATASET_ROOT)" --hard-dataset-id "$(HARD_DATASET_ID)" --request "$(MINING_REQUEST)"

hard-import:
	$(if $(strip $(HARD_DATASET_ID)),,$(error HARD_DATASET_ID is required))
	$(CLI) import-hard-review --dataset-root "$(HAND_DATASET_ROOT)" --hard-dataset-id "$(HARD_DATASET_ID)"

hard-publish:
	$(if $(strip $(HARD_DATASET_ID)),,$(error HARD_DATASET_ID is required))
	$(CLI) publish-hard-review --dataset-root "$(HAND_DATASET_ROOT)" --hard-dataset-id "$(HARD_DATASET_ID)"

registry-check:
	$(CLI) registry-check --dataset-root "$(HAND_DATASET_ROOT)"

compile:
	$(PYTHON) -B -c "from pathlib import Path; files=[p for root in ('hand_autolabel','scripts','tests','tools') for p in Path(root).rglob('*.py')]; [compile(p.read_bytes(), str(p), 'exec') for p in files]; print('syntax-checked', len(files), 'Python files')"

test:
	$(PYTHON) -B -m unittest discover -s tests -p "test_*.py"
