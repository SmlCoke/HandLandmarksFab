# Makefile for HandLandmarkerFab

# 默认配置文件路径
# smoke test use config, running in local machine
CONFIG ?= configs/autolabel.yaml
# training, validate, test set formal half-automatic annotation use following config:
TRAIN_CONFIG ?= configs/autolabel_train.yaml
VALIDATE_CONFIG ?= configs/autolabel_val.yaml
TEST_CONFIG ?= configs/autolabel_test.yaml

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
	@echo "  make validate_images_val          run image validation for validation set"
	@echo "  make validate_images_test         run image validation for test set"
	@echo "  make palm_detection_smoke         run palm detection (smoke test)"
	@echo "  make palm_detection_train         run palm detection for training set"
	@echo "  make palm_detection_val           run palm detection for validation set"
	@echo "  make palm_detection_test          run palm detection for test set"
	@echo "  make build_roi_smoke              build Hand ROI crops (smoke test)"
	@echo "  make build_roi_train              build Hand ROI crops for training set"
	@echo "  make build_roi_val                build Hand ROI crops for validation set"
	@echo "  make build_roi_test               build Hand ROI crops for test set"
	@echo "  make run_mediapipe_smoke          run MediaPipe on ROI (smoke test)"
	@echo "  make run_mediapipe_train          run MediaPipe on ROI for training set"
	@echo "  make run_mediapipe_val            run MediaPipe on ROI for validation set"
	@echo "  make run_mediapipe_test           run MediaPipe on ROI for test set"
	@echo "  make export_cvat_smoke            export CVAT XML (smoke test)"
	@echo "  make export_cvat_train            export CVAT XML for training set"
	@echo "  make export_cvat_val              export CVAT XML for validation set"
	@echo "  make export_cvat_test             export CVAT XML for test set"
	@echo "  make import_cvat_smoke            import CVAT review results (smoke test)"
	@echo "  make import_cvat_train            import CVAT review results for training set"
	@echo "  make import_cvat_val              import CVAT review results for validation set"
	@echo "  make import_cvat_test             import CVAT review results for test set"
	@echo "  make visualize_smoke              visualize annotations (smoke test)"
	@echo "  make visualize_train              visualize annotations for training set"
	@echo "  make visualize_val                visualize annotations for validation set"
	@echo "  make visualize_test               visualize annotations for test set"
	@echo "  make finalize_smoke               generate final labels (smoke test)"
	@echo "  make finalize_train               generate final training labels"
	@echo "  make finalize_val                 generate final validation labels"
	@echo "  make finalize_test                generate final test labels"
	@echo ""
	@echo "Variable overrides:"
	@echo "  make palm_detection_smoke CONFIG=path/to/config.yaml"

# ----- scripts flow -----

# 00_validate_images.py
validate_images_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(CONFIG)

validate_images_train:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(TRAIN_CONFIG)

validate_images_val:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(VALIDATE_CONFIG)

validate_images_test:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(TEST_CONFIG)

## 01_export_palm_detections.py
palm_detection_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(CONFIG)

palm_detection_train:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(TRAIN_CONFIG)

palm_detection_val:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(VALIDATE_CONFIG)

palm_detection_test:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(TEST_CONFIG)

## 02_build_hand_roi_crops.py
build_roi_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(CONFIG)

build_roi_train:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(TRAIN_CONFIG)

build_roi_val:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(VALIDATE_CONFIG)

build_roi_test:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(TEST_CONFIG)


## 03_run_mediapipe_on_rois.py
run_mediapipe_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(CONFIG)

run_mediapipe_train:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(TRAIN_CONFIG)

run_mediapipe_val:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(VALIDATE_CONFIG)

run_mediapipe_test:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(TEST_CONFIG)

## 04_export_cvat_xml.py
export_cvat_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(CONFIG)

export_cvat_train:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(TRAIN_CONFIG)

export_cvat_val:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(VALIDATE_CONFIG)

export_cvat_test:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(TEST_CONFIG)

## 05_import_cvat_xml.py
import_cvat_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(CONFIG)

import_cvat_train:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(TRAIN_CONFIG)

import_cvat_val:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(VALIDATE_CONFIG)

import_cvat_test:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(TEST_CONFIG)

## 06_visualize_autolabels.py
visualize_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(CONFIG)

visualize_train:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(TRAIN_CONFIG)

visualize_val:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(VALIDATE_CONFIG)

visualize_test:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(TEST_CONFIG)

## 07_finalize_training_labels.py
finalize_smoke:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(CONFIG)

finalize_train:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(TRAIN_CONFIG)

finalize_val:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(VALIDATE_CONFIG)

finalize_test:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(TEST_CONFIG)
