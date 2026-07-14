# Makefile for HandLandmarkerFab

# Optional per-machine overrides. Makefile.local is intentionally not committed.
-include Makefile.local

# Root containing vals_data, vali_data, and test_data. A command-line value or
# an external environment variable takes precedence over this default.
HAND_DATA_ROOT ?= ../autodl-tmp/TrainFab/HLML-2.0

# Set to 1/true/yes/on to render landmarks on 02_roi_crops/images after stage 03.
VISUALIZE_MEDIAPIPE_ROIS ?= 0

# Set to 1/true/yes/on to render every canonical ROI after 07A train finalization.
VISUALIZE_FINALIZED_TRAIN_ROIS ?= 0

# 默认配置文件路径
# smoke test use config, running in local machine
CONFIG ?= configs/autolabel.yaml
# training, validate, test set formal half-automatic annotation use following config:
TRAIN_CONFIG ?= configs/autolabel_train.yaml
VAL_SHARED_CONFIG ?= configs/autolabel_val.yaml
VAL_INDEPENDENT_CONFIG ?= configs/autolabel_vali.yaml
TEST_CONFIG ?= configs/autolabel_test.yaml
FINALIZE_TRAIN_CONFIG ?= configs/finalize_train.yaml
FINALIZE_VAL_CONFIG ?= configs/finalize_val.yaml
FINALIZE_TEST_CONFIG ?= configs/finalize_test.yaml

# Python 解释器（如有需要可改为 python3）
PYTHON = python

# 所有脚本的公共前缀
SCRIPTS_DIR = scripts

# ----- 目标定义 -----

# 默认目标（直接执行 make 时触发）
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make validate_images_smoke        run image validation (smoke test)"
	@echo "  make validate_images_train        run image validation for training set"
	@echo "  make validate_images_vals         validate shared validation images"
	@echo "  make validate_images_vali         validate independent validation images"
	@echo "  make validate_images_test         run image validation for test set"
	@echo "  make palm_detection_smoke         run palm detection (smoke test)"
	@echo "  make palm_detection_train         run palm detection for training set"
	@echo "  make palm_detection_vals          run Palm on shared validation set"
	@echo "  make palm_detection_vali          run Palm on independent validation set"
	@echo "  make palm_detection_test          run palm detection for test set"
	@echo "  make build_roi_smoke              build Hand ROI crops (smoke test)"
	@echo "  make build_roi_train              build Hand ROI crops for training set"
	@echo "  make build_roi_vals               build ROI for shared validation set"
	@echo "  make build_roi_vali               build ROI for independent validation set"
	@echo "  make build_roi_test               build Hand ROI crops for test set"
	@echo "  make run_mediapipe_smoke          run MediaPipe on ROI (smoke test)"
	@echo "  make run_mediapipe_train          run MediaPipe on ROI for training set"
	@echo "  make run_mediapipe_vals           run MediaPipe on shared validation ROI"
	@echo "  make run_mediapipe_vali           run MediaPipe on independent validation ROI"
	@echo "  make run_mediapipe_test           run MediaPipe on ROI for test set"
	@echo "  make export_cvat_smoke            export CVAT XML (smoke test)"
	@echo "  make export_cvat_train            export CVAT XML for training set"
	@echo "  make export_cvat_vals             export shared validation CVAT XML"
	@echo "  make export_cvat_vali             export independent validation CVAT XML"
	@echo "  make export_cvat_test             export CVAT XML for test set"
	@echo "  make import_cvat_smoke            import CVAT review results (smoke test)"
	@echo "  make import_cvat_train            import CVAT review results for training set"
	@echo "  make import_cvat_vals             import shared validation CVAT review"
	@echo "  make import_cvat_vali             import independent validation CVAT review"
	@echo "  make import_cvat_test             import CVAT review results for test set"
	@echo "  make visualize_smoke              visualize annotations (smoke test)"
	@echo "  make visualize_train              visualize annotations for training set"
	@echo "  make visualize_vals               visualize shared validation annotations"
	@echo "  make visualize_vali               visualize independent validation annotations"
	@echo "  make visualize_test               visualize annotations for test set"
	@echo "  make finalize_train_pretrain      07A: generate pseudo-label pretraining set"
	@echo "  make finalize_train_finetune      07A: generate Gold+pseudo fine-tuning set"
	@echo "  make finalize_val                 07B: freeze strict validation Gold labels"
	@echo "  make finalize_test                07B: freeze strict test Gold labels"
	@echo ""
	@echo "Variable overrides:"
	@echo "  make palm_detection_smoke CONFIG=path/to/config.yaml"
	@echo "  make validate_images_vals HAND_DATA_ROOT=/path/to/data-root"
	@echo "  make run_mediapipe_train VISUALIZE_MEDIAPIPE_ROIS=1"
	@echo "  make finalize_train_pretrain VISUALIZE_FINALIZED_TRAIN_ROIS=1"
	@echo "  For a persistent local value, copy Makefile.local.example to Makefile.local"

# ----- scripts flow -----

# 00_validate_images.py
validate_images_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(CONFIG)

validate_images_train:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(TRAIN_CONFIG)

validate_images_vals:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(VAL_SHARED_CONFIG)

validate_images_vali:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(VAL_INDEPENDENT_CONFIG)

validate_images_test:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(TEST_CONFIG)

## 01_export_palm_detections.py
palm_detection_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(CONFIG)

palm_detection_train:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(TRAIN_CONFIG)

palm_detection_vals:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(VAL_SHARED_CONFIG)

palm_detection_vali:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(VAL_INDEPENDENT_CONFIG)

palm_detection_test:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(TEST_CONFIG)

## 02_build_hand_roi_crops.py
build_roi_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(CONFIG)

build_roi_train:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(TRAIN_CONFIG)

build_roi_vals:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(VAL_SHARED_CONFIG)

build_roi_vali:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(VAL_INDEPENDENT_CONFIG)

build_roi_test:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(TEST_CONFIG)


## 03_run_mediapipe_on_rois.py
run_mediapipe_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

run_mediapipe_train:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(TRAIN_CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

run_mediapipe_vals:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(VAL_SHARED_CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

run_mediapipe_vali:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(VAL_INDEPENDENT_CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

run_mediapipe_test:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(TEST_CONFIG) --visualize-rois $(VISUALIZE_MEDIAPIPE_ROIS)

## 04_export_cvat_xml.py
export_cvat_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(CONFIG)

export_cvat_train:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(TRAIN_CONFIG)

export_cvat_vals:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(VAL_SHARED_CONFIG)

export_cvat_vali:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(VAL_INDEPENDENT_CONFIG)

export_cvat_test:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(TEST_CONFIG)

## 05_import_cvat_xml.py
import_cvat_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(CONFIG)

import_cvat_train:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(TRAIN_CONFIG)

import_cvat_vals:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(VAL_SHARED_CONFIG)

import_cvat_vali:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(VAL_INDEPENDENT_CONFIG)

import_cvat_test:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(TEST_CONFIG)

## 06_visualize_autolabels.py
visualize_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(CONFIG)

visualize_train:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(TRAIN_CONFIG)

visualize_vals:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(VAL_SHARED_CONFIG)

visualize_vali:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(VAL_INDEPENDENT_CONFIG)

visualize_test:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(TEST_CONFIG)

## 07A/07B finalizers
finalize_train_pretrain:
	$(PYTHON) $(SCRIPTS_DIR)/07A_finalize_training_labels.py --config $(FINALIZE_TRAIN_CONFIG) --stage pretrain --visualize-rois $(VISUALIZE_FINALIZED_TRAIN_ROIS)

finalize_train_finetune:
	$(PYTHON) $(SCRIPTS_DIR)/07A_finalize_training_labels.py --config $(FINALIZE_TRAIN_CONFIG) --stage finetune --visualize-rois $(VISUALIZE_FINALIZED_TRAIN_ROIS)

finalize_val:
	$(PYTHON) $(SCRIPTS_DIR)/07B_finalize_evaluation_labels.py --config $(FINALIZE_VAL_CONFIG) --split val

finalize_test:
	$(PYTHON) $(SCRIPTS_DIR)/07B_finalize_evaluation_labels.py --config $(FINALIZE_TEST_CONFIG) --split test
