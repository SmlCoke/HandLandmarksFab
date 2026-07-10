# Makefile for HandLandmarkerFab

# 默认配置文件路径
CONFIG ?= configs/autolabel.yaml
TEST_CONFIG ?= configs/autolabel_test_runtime.yaml
VALIDATE_CONFIG ?= configs/autolabel_val_runtime.yaml

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
	@echo "  make validate_images         run image validation for training set"
	@echo "  make val_validate_images     run image validation for validation set"
	@echo "  make test_validate_images    run image validation for test set"
	@echo "  make palm_detection          run palm detection for training set"
	@echo "  make val_palm_detection      run palm detection for validation set"
	@echo "  make test_palm_detection     run palm detection for test set"
	@echo "  make build_roi               build Hand ROI crops for training set"
	@echo "  make val_build_roi           build Hand ROI crops for validation set"
	@echo "  make test_build_roi          build Hand ROI crops for test set"
	@echo "  make run_mediapipe           run MediaPipe on ROI for training set"
	@echo "  make val_run_mediapipe       run MediaPipe on ROI for validation set"
	@echo "  make test_run_mediapipe      run MediaPipe on ROI for test set"
	@echo "  make export_cvat             export CVAT XML for training set"
	@echo "  make val_export_cvat         export CVAT XML for validation set"
	@echo "  make test_export_cvat        export CVAT XML for test set"
	@echo "  make import_cvat             import CVAT review results for training set"
	@echo "  make val_import_cvat         import CVAT review results for validation set"
	@echo "  make test_import_cvat        import CVAT review results for test set"
	@echo "  make visualize               visualize annotations for training set"
	@echo "  make val_visualize           visualize annotations for validation set"
	@echo "  make test_visualize          visualize annotations for test set"
	@echo "  make finalize                generate final training labels"
	@echo "  make val_finalize            generate final validation labels"
	@echo "  make test_finalize           generate final test labels"
	@echo ""
	@echo "Variable overrides:"
	@echo "  make palm_detection CONFIG=path/to/config.yaml"

# ----- scripts flow -----

# 00_validate_images.py
validate_images:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(CONFIG)

val_validate_images:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(VALIDATE_CONFIG)

test_validate_images:
	$(PYTHON) $(SCRIPTS_DIR)/00_validate_images.py --config $(TEST_CONFIG)

## 01_export_palm_detections.py
palm_detection:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(CONFIG)

val_palm_detection:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(VALIDATE_CONFIG)

test_palm_detection:
	$(PYTHON) $(SCRIPTS_DIR)/01_export_palm_detections.py --config $(TEST_CONFIG)

## 02_build_hand_roi_crops.py
build_roi:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(CONFIG)

val_build_roi:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(VALIDATE_CONFIG)

test_build_roi:
	$(PYTHON) $(SCRIPTS_DIR)/02_build_hand_roi_crops.py --config $(TEST_CONFIG)


## 03_run_mediapipe_on_rois.py
run_mediapipe:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(CONFIG)

val_run_mediapipe:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(VALIDATE_CONFIG)

test_run_mediapipe:
	$(PYTHON) $(SCRIPTS_DIR)/03_run_mediapipe_on_rois.py --config $(TEST_CONFIG)

## 04_export_cvat_xml.py
export_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(CONFIG)

val_export_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(VALIDATE_CONFIG)

test_export_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/04_export_cvat_xml.py --config $(TEST_CONFIG)

## 05_import_cvat_xml.py
import_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(CONFIG)

val_import_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(VALIDATE_CONFIG)

test_import_cvat:
	$(PYTHON) $(SCRIPTS_DIR)/05_import_cvat_xml.py --config $(TEST_CONFIG)

## 06_visualize_autolabels.py
visualize:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(CONFIG)

test_visualize:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(TEST_CONFIG)

val_visualize:
	$(PYTHON) $(SCRIPTS_DIR)/06_visualize_autolabels.py --config $(VALIDATE_CONFIG)

## 07_finalize_training_labels.py
finalize:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(CONFIG)

val_finalize:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(VALIDATE_CONFIG)

test_finalize:
	$(PYTHON) $(SCRIPTS_DIR)/07_finalize_training_labels.py --config $(TEST_CONFIG)
