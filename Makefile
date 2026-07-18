# HLMF 2.0: one source pipeline, plus dataset finalization and Gold workflows.

-include Makefile.local

PYTHON ?= python
HAND_WORK_ROOT ?= /root/autodl-tmp/TrainFab/HLML-3.0
HAND_DATASET_ROOT ?= /root/autodl-tmp/DatesetFab
HLMF_SOURCE_ROOT ?= data
HAND_FINETUNE_ID ?= v3-finetune-r1
HAND_PRETRAIN_ID ?= v3-pretrain-r1
DRAGON_SOURCE_ROOT ?=
DRAGON_BATCH_ID ?=
export HAND_WORK_ROOT HAND_DATASET_ROOT HLMF_SOURCE_ROOT
export HAND_FINETUNE_ID HAND_PRETRAIN_ID

AUTOLABEL_CONFIG ?= configs/autolabel.yaml
AUTOLABEL_ROLE ?= train
AUTOLABEL_OVERRIDES ?= {}
export AUTOLABEL_ROLE AUTOLABEL_OVERRIDES
FINALIZE_TRAIN_CONFIG ?= configs/finalize_train.yaml
FINALIZE_VAL_CONFIG ?= configs/finalize_val.yaml
FINALIZE_TEST_CONFIG ?= configs/finalize_test.yaml
FINETUNE_GOLD_CONFIG ?= configs/finetune_gold.yaml
FINALIZE_FINETUNE_CONFIG ?= configs/finalize_finetune.yaml
DRAGON_GOLD_CONFIG ?= configs/dragon_gold.yaml

FINETUNE_SOURCE_ID ?=
FINETUNE_SOURCE_MODE ?=
FINETUNE_RAW_SOURCE_ROOT ?=
FINETUNE_SELECTION_REQUEST ?=
FINETUNE_MAX_ITEMS ?=
BASE_FINETUNE_ID ?=
VISUALIZE_MEDIAPIPE_ROIS ?= 0
VISUALIZE_FINALIZED_TRAIN_ROIS ?= 0

.DEFAULT_GOAL := help
.PHONY: help paths autolabel validate_images palm_detection build_roi run_mediapipe \
	export_cvat import_cvat visualize finalize_train_pretrain finalize_val finalize_test \
	prepare_dragon_gold build_pretrain_source_registry seed_finetune_gold \
	export_finetune_gold import_finetune_gold finalize_train_finetune compile test

help:
	@echo HLMF 2.0 - one configurable source pipeline
	@echo   make paths HLMF_SOURCE_ROOT=/path/to/source
	@echo   make autolabel HLMF_SOURCE_ROOT=... AUTOLABEL_ROLE=train/val/test AUTOLABEL_OVERRIDES='{"palm":{"negative_candidate_threshold":0.2}}'
	@echo   make validate_images palm_detection build_roi run_mediapipe
	@echo   make export_cvat import_cvat visualize
	@echo   make finalize_train_pretrain build_pretrain_source_registry
	@echo   make prepare_dragon_gold DRAGON_SOURCE_ROOT=... DRAGON_BATCH_ID=...
	@echo   make seed_finetune_gold BASE_FINETUNE_ID=... HAND_FINETUNE_ID=...
	@echo   make export_finetune_gold FINETUNE_SOURCE_ID=... FINETUNE_SOURCE_MODE=...
	@echo   make import_finetune_gold FINETUNE_SOURCE_ID=...
	@echo   make finalize_train_finetune
	@echo   make compile test

paths:
	@echo HAND_DATASET_ROOT=$(HAND_DATASET_ROOT)
	@echo HAND_WORK_ROOT=$(HAND_WORK_ROOT)
	@echo HLMF_SOURCE_ROOT=$(HLMF_SOURCE_ROOT)
	@echo HAND_PRETRAIN_ID=$(HAND_PRETRAIN_ID)
	@echo HAND_FINETUNE_ID=$(HAND_FINETUNE_ID)
	@echo AUTOLABEL_ROLE=$(AUTOLABEL_ROLE)
	@echo AUTOLABEL_OVERRIDES=$(AUTOLABEL_OVERRIDES)

autolabel: validate_images palm_detection build_roi run_mediapipe

validate_images:
	$(PYTHON) scripts/00_validate_images.py --config $(AUTOLABEL_CONFIG)

palm_detection:
	$(PYTHON) scripts/01_export_palm_detections.py --config $(AUTOLABEL_CONFIG)

build_roi:
	$(PYTHON) scripts/02_build_hand_roi_crops.py --config $(AUTOLABEL_CONFIG)

run_mediapipe:
	$(PYTHON) scripts/03_run_mediapipe_on_rois.py --config $(AUTOLABEL_CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

export_cvat:
	$(PYTHON) scripts/04_export_cvat_xml.py --config $(AUTOLABEL_CONFIG)

import_cvat:
	$(PYTHON) scripts/05_import_cvat_xml.py --config $(AUTOLABEL_CONFIG)

visualize:
	$(PYTHON) scripts/06_visualize_autolabels.py --config $(AUTOLABEL_CONFIG)

finalize_train_pretrain:
	$(PYTHON) scripts/07A_finalize_training_labels.py --config $(FINALIZE_TRAIN_CONFIG) --stage pretrain --visualize-rois $(VISUALIZE_FINALIZED_TRAIN_ROIS)

finalize_val:
	$(PYTHON) scripts/07B_finalize_evaluation_labels.py --config $(FINALIZE_VAL_CONFIG) --split val

finalize_test:
	$(PYTHON) scripts/07B_finalize_evaluation_labels.py --config $(FINALIZE_TEST_CONFIG) --split test

prepare_dragon_gold:
	$(if $(strip $(DRAGON_SOURCE_ROOT)),,$(error DRAGON_SOURCE_ROOT is required))
	$(if $(strip $(DRAGON_BATCH_ID)),,$(error DRAGON_BATCH_ID is required))
	$(PYTHON) scripts/08_finetune_gold.py prepare-dragon --config $(DRAGON_GOLD_CONFIG) --raw-source-root "$(DRAGON_SOURCE_ROOT)" --batch-id "$(DRAGON_BATCH_ID)"

build_pretrain_source_registry:
	$(PYTHON) scripts/08_finetune_gold.py source-registry --config $(FINALIZE_TRAIN_CONFIG)

seed_finetune_gold:
	$(if $(strip $(BASE_FINETUNE_ID)),,$(error BASE_FINETUNE_ID is required))
	$(PYTHON) scripts/08_finetune_gold.py seed --config $(FINETUNE_GOLD_CONFIG) --base-finetune-id $(BASE_FINETUNE_ID) --finetune-id $(HAND_FINETUNE_ID)

export_finetune_gold:
	$(if $(strip $(FINETUNE_SOURCE_ID)),,$(error FINETUNE_SOURCE_ID is required))
	$(if $(strip $(FINETUNE_SOURCE_MODE)),,$(error FINETUNE_SOURCE_MODE is required))
	$(PYTHON) scripts/08_finetune_gold.py export --config $(FINETUNE_GOLD_CONFIG) --source-id $(FINETUNE_SOURCE_ID) --source-mode $(FINETUNE_SOURCE_MODE) $(if $(strip $(FINETUNE_RAW_SOURCE_ROOT)),--raw-source-root $(FINETUNE_RAW_SOURCE_ROOT),) $(if $(strip $(FINETUNE_SELECTION_REQUEST)),--selection-request $(FINETUNE_SELECTION_REQUEST),) $(if $(strip $(FINETUNE_MAX_ITEMS)),--max-items $(FINETUNE_MAX_ITEMS),)

import_finetune_gold:
	$(PYTHON) scripts/08_finetune_gold.py import --config $(FINETUNE_GOLD_CONFIG) $(if $(strip $(FINETUNE_SOURCE_ID)),--source-id $(FINETUNE_SOURCE_ID),--all)

finalize_train_finetune:
	$(PYTHON) scripts/08_finetune_gold.py finalize --config $(FINALIZE_FINETUNE_CONFIG)

compile:
	$(PYTHON) -c "from pathlib import Path; files=[p for root in ('hand_autolabel','scripts','tests') for p in Path(root).rglob('*.py')]; [compile(p.read_bytes(), str(p), 'exec') for p in files]; print('syntax-checked', len(files), 'Python files')"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
