from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hand_autolabel.dataset_v3 import (
    DatasetContractError,
    ROI_CONTRACT_VERSION,
    WarehouseRegistry,
    apply_label_provenance,
    assert_palm_capture_distance_supported,
    clean_variant_visualizations,
    delete_source_variant,
    enrich_palm_rows,
    enrich_roi_rows,
    parse_capture_source_id,
    palm_capture_distance_policy,
    prepare_negative_review,
    prepare_selection_review,
    proposal_paths,
    publish_negative_review,
    publish_selection_review,
    source_root,
    validate_and_normalize_source,
)
from hand_autolabel.formats import load_yaml_config, read_jsonl, write_json, write_jsonl
from hand_autolabel.gold_reviews import (
    import_hard_review,
    prepare_hard_review,
    publish_hard_review,
)
from hand_autolabel.mediapipe_roi_visualization import (
    TrainingRoiVisualizationError,
    evenly_spaced_sample,
    render_autolabel_roi_visualizations,
    render_original_image_visualizations,
)
from hand_autolabel.progress import track_progress
from scripts.hlmf import (
    _dataset_manifest,
    _load_public_configs,
    _parser,
    _partition_labels,
    _quality_gate_rejection_counts,
    _run_delete_source_variant,
    _run_existing_original_image_visualization,
    _run_existing_roi_visualization,
    _run_publish_source,
    _run_source_pipeline,
    _validate_evaluation_limits,
)
from tools.downsample import downsample


CAPTURE_TRAIN = "white-mid-bright-fist-train-s01-peak"
CAPTURE_VAL = "complex-far-dark-cross-val-s02-dragon"


def write_tiff(path: Path, shape: tuple[int, int] = (1280, 720), value: int = 31) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]):
        raise RuntimeError(path)


class DatasetV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _source(self, capture: str = CAPTURE_TRAIN, scope: str = "pretrain") -> Path:
        return source_root(self.root, scope, "national-r1", capture)

    def _validated_source(self) -> tuple[Path, dict]:
        source = self._source()
        write_tiff(source / "images" / "frame001.tiff")
        report = validate_and_normalize_source(
            self.root, "pretrain", "national-r1", CAPTURE_TRAIN
        )
        return source, report

    def _registered_roi(self) -> tuple[Path, dict]:
        source, _ = self._validated_source()
        raw = read_jsonl(source / "raw_images.jsonl")[0]
        crop = source / "02_roi_crops" / "p01" / "images" / "crop.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop), np.zeros((256, 256), dtype=np.uint8))
        roi = {
            "roi_id": "roi_test001",
            "raw_image_id": raw["raw_image_id"],
            "capture_source_id": CAPTURE_TRAIN,
            "dataset_id": "national-r1",
            "split": "train",
            "proposal_variant": "p01",
            "proposal_slot": 0,
            "proposal_kind": "negative_candidate",
            "crop_relpath": str(crop.relative_to(self.root)).replace("\\", "/"),
            "hand_presence": {"present": False},
        }
        WarehouseRegistry(self.root).register_rois([roi])
        return crop, roi

    def test_capture_source_contract_and_scope(self) -> None:
        parsed = parse_capture_source_id(CAPTURE_TRAIN)
        self.assertEqual(parsed["performer"], "peak")
        self.assertEqual(parsed["split"], "train")
        self.assertEqual(
            self.root / "PretrainSource" / "FullEnhance0801" / CAPTURE_TRAIN,
            source_root(self.root, "pretrain", "FullEnhance0801", CAPTURE_TRAIN),
        )
        with self.assertRaises(DatasetContractError):
            parse_capture_source_id("white-mid-bright-two-hands-train-s01-peak")
        with self.assertRaises(DatasetContractError):
            source_root(self.root, "pretrain", "national-r1", CAPTURE_VAL)
        self.assertEqual(
            self.root
            / "GoldSource"
            / "ReviewedDatasets"
            / "gold-r1"
            / CAPTURE_TRAIN,
            source_root(self.root, "gold", "gold-r1", CAPTURE_TRAIN),
        )
        with self.assertRaises(DatasetContractError):
            source_root(self.root, "gold", "gold-r1", CAPTURE_VAL)

    def test_palm_capture_distance_policy_accepts_near_mid_and_rejects_other_distances(self) -> None:
        cfg = {
            "palm": {
                "model_id": "eos-2.0",
                "supported_capture_distances": ["near", "mid"],
            }
        }
        near = palm_capture_distance_policy(
            "white-near-bright-fist-train-s01-peak", cfg
        )
        mid = assert_palm_capture_distance_supported(CAPTURE_TRAIN, cfg)
        self.assertTrue(near["supported"])
        self.assertTrue(mid["supported"])
        self.assertEqual(["near", "mid"], near["supported_capture_distances"])
        for capture_id in (
            CAPTURE_VAL,
            "white-unknown-bright-fist-train-s01-peak",
        ):
            with self.assertRaisesRegex(
                DatasetContractError,
                r"unsupported_capture_distance:model=eos-2\.0:distance=.*:supported=near,mid",
            ):
                assert_palm_capture_distance_supported(capture_id, cfg)

    def test_palm_capture_distance_policy_rejects_invalid_config(self) -> None:
        bad_values = (
            {},
            {"palm": {}},
            {"palm": {"model_id": "eos-2.0"}},
            {
                "palm": {
                    "model_id": "eos-2.0",
                    "supported_capture_distances": "near,mid",
                }
            },
            {
                "palm": {
                    "model_id": "eos-2.0",
                    "supported_capture_distances": [],
                }
            },
            {
                "palm": {
                    "model_id": "eos-2.0",
                    "supported_capture_distances": ["near", "Near"],
                }
            },
            {
                "palm": {
                    "model_id": "eos-2.0",
                    "supported_capture_distances": ["near", "near"],
                }
            },
        )
        for cfg in bad_values:
            with self.subTest(cfg=cfg), self.assertRaises(DatasetContractError):
                palm_capture_distance_policy(CAPTURE_TRAIN, cfg)

    def test_far_pipeline_and_publish_fail_before_creating_variant_outputs(self) -> None:
        cfg = {
            "palm": {
                "model_id": "eos-2.0",
                "supported_capture_distances": ["near", "mid"],
            }
        }
        args = SimpleNamespace(
            dataset_root=str(self.root),
            scope="eval",
            dataset_id="national-r1",
            capture_source_id=CAPTURE_VAL,
            proposal_variant="eos-2.0-r1",
        )
        source = source_root(self.root, "eval", "national-r1", CAPTURE_VAL)
        with patch("scripts.hlmf._run_validate") as validate:
            with self.assertRaisesRegex(
                DatasetContractError, "unsupported_capture_distance"
            ):
                _run_source_pipeline(args, cfg, evaluation=True)
            validate.assert_not_called()
        with self.assertRaisesRegex(
            DatasetContractError, "unsupported_capture_distance"
        ):
            _run_publish_source(args, cfg)
        paths = proposal_paths(source, args.proposal_variant)
        self.assertFalse(any(path.exists() for key, path in paths.items() if key != "images"))

    def test_check_palm_distance_parser_requires_only_capture_source(self) -> None:
        args = _parser().parse_args(
            ["check-palm-distance", "--capture-source-id", CAPTURE_TRAIN]
        )
        self.assertEqual("check-palm-distance", args.command)
        self.assertEqual(CAPTURE_TRAIN, args.capture_source_id)

    def test_validate_rotates_once_and_is_idempotent(self) -> None:
        source = self._source()
        write_tiff(source / "images" / "frame001.tiff", shape=(1280, 720))
        first = validate_and_normalize_source(
            self.root, "pretrain", "national-r1", CAPTURE_TRAIN
        )
        second = validate_and_normalize_source(
            self.root, "pretrain", "national-r1", CAPTURE_TRAIN
        )
        image = cv2.imread(str(source / "images" / "frame001.tiff"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape, (720, 1280))
        self.assertEqual(first["rotated_clockwise"], 1)
        self.assertEqual(second["rotated_clockwise"], 0)
        self.assertEqual(
            read_jsonl(source / "raw_images.jsonl")[0]["raw_image_id"],
            read_jsonl(source / "raw_images.jsonl")[0]["raw_image_id"],
        )

    def test_uint16_rotation_preserves_pixels_losslessly(self) -> None:
        source = source_root(self.root, "pretrain", "uint16-r1", CAPTURE_TRAIN)
        original = np.arange(1280 * 720, dtype=np.uint16).reshape(1280, 720)
        path = source / "images" / "frame.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), original, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]))
        validate_and_normalize_source(self.root, "pretrain", "uint16-r1", CAPTURE_TRAIN)
        rotated = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertEqual(np.uint16, rotated.dtype)
        np.testing.assert_array_equal(cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE), rotated)

    def test_validate_rejects_wrong_size_and_non_tiff(self) -> None:
        source = self._source()
        write_tiff(source / "images" / "bad.tiff", shape=(100, 100))
        with self.assertRaises(DatasetContractError):
            validate_and_normalize_source(self.root, "pretrain", "national-r1", CAPTURE_TRAIN)
        self.assertTrue((source / "qc" / "image_validation_report.json").is_file())
        (source / "images" / "bad.tiff").unlink()
        cv2.imwrite(str(source / "images" / "bad.png"), np.zeros((720, 1280), dtype=np.uint8))
        with self.assertRaises(DatasetContractError):
            validate_and_normalize_source(self.root, "pretrain", "national-r1", CAPTURE_TRAIN)
        report = json.loads(
            (source / "qc" / "image_validation_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual("non_tiff_input", report["errors"][0]["error"])

    def test_raw_id_survives_unique_rename(self) -> None:
        source, _ = self._validated_source()
        before = read_jsonl(source / "raw_images.jsonl")[0]["raw_image_id"]
        (source / "images" / "frame001.tiff").rename(source / "images" / "renamed.tiff")
        report = validate_and_normalize_source(
            self.root, "pretrain", "national-r1", CAPTURE_TRAIN
        )
        after = read_jsonl(source / "raw_images.jsonl")[0]["raw_image_id"]
        self.assertEqual(before, after)
        self.assertEqual(report["raw_ids_reused_after_rename"], 1)

    def test_proposal_variants_share_raw_id_but_not_roi_id(self) -> None:
        source, _ = self._validated_source()
        raw = read_jsonl(source / "raw_images.jsonl")
        backend = [
            {
                "image": "frame001.tiff",
                "detections": [
                    {
                        "bbox_norm": [0.2, 0.2, 0.8, 0.8],
                        "keypoints_norm": {"p0": [0.4, 0.7], "p9": [0.5, 0.3]},
                        "score": 0.9,
                        "valid": True,
                    }
                ],
                "negative_candidates": [],
            }
        ]
        p01 = enrich_palm_rows(raw, backend, "p01")
        p02 = enrich_palm_rows(raw, backend, "p02")
        self.assertEqual(p01[0]["raw_image_id"], p02[0]["raw_image_id"])
        base = {
            "crop_id": "temporary",
            "image": "frame001.tiff",
            "palm_det_id": p01[0]["detections"][0]["palm_det_id"],
            "crop_path": str(source / "02_roi_crops" / "p01" / "images" / "x.png"),
        }
        roi01 = enrich_roi_rows([base], p01, self.root, "p01")[0]
        base02 = dict(base, palm_det_id=p02[0]["detections"][0]["palm_det_id"])
        roi02 = enrich_roi_rows([base02], p02, self.root, "p02")[0]
        self.assertNotEqual(roi01["roi_id"], roi02["roi_id"])
        self.assertEqual(roi01["roi_contract_version"], ROI_CONTRACT_VERSION)

    def test_runtime_and_negative_candidates_share_one_slot_namespace(self) -> None:
        source, _ = self._validated_source()
        raw = read_jsonl(source / "raw_images.jsonl")
        base = {
            "bbox_norm": [0.2, 0.2, 0.8, 0.8],
            "keypoints_norm": {"p0": [0.4, 0.7], "p9": [0.5, 0.3]},
            "score": 0.9,
            "valid": True,
        }
        rows = enrich_palm_rows(
            raw,
            [
                {
                    "image": "frame001.tiff",
                    "detections": [dict(base)],
                    "negative_candidates": [dict(base, score=0.2)],
                }
            ],
            "p01",
        )
        proposals = rows[0]["detections"] + rows[0]["negative_candidates"]
        self.assertEqual([0, 1], sorted(item["proposal_slot"] for item in proposals))
        self.assertEqual(2, len({item["palm_det_id"] for item in proposals}))

    def test_label_provenance_distinguishes_teacher_and_human_changes(self) -> None:
        points = [{"id": index, "x": index / 30.0, "y": index / 40.0} for index in range(21)]
        draft = {
            "roi_id": "roi_1",
            "crop_id": "roi_1",
            "hand_presence": {"present": True},
            "landmarks_crop_norm": points,
        }
        unchanged = apply_label_provenance([draft], {"roi_1": draft}, True)[0]
        corrected = dict(draft, landmarks_crop_norm=[dict(point) for point in points])
        corrected["landmarks_crop_norm"][4]["x"] += 0.1
        changed = apply_label_provenance([corrected], {"roi_1": draft}, True)[0]
        abstain = apply_label_provenance(
            [dict(draft, roi_id="roi_2", crop_id="roi_2")],
            {"roi_2": {"hand_presence": {"present": False}}},
            True,
        )[0]
        self.assertEqual(unchanged["label_origin"], "mediapipe")
        self.assertEqual(changed["label_origin"], "mediapipe_human_corrected")
        self.assertEqual(changed["human_modified_landmark_ids"], [4])
        self.assertEqual(abstain["label_origin"], "human")
        self.assertTrue(abstain["human_modified_presence"])
        self.assertEqual(
            "mediapipe_hand_landmarker_task",
            unchanged["hand_presence_teacher_model_id"],
        )

        rtmpose_draft = dict(
            draft,
            source="rtmpose_m_hand5_onnx",
            handedness_teacher_model_id="hand-classifier-handedness-handpresence-0807",
            hand_presence_teacher_model_id="hand-classifier-handedness-handpresence-0807",
        )
        rtmpose_unchanged = apply_label_provenance(
            [rtmpose_draft], {"roi_1": rtmpose_draft}, True
        )[0]
        rtmpose_corrected = dict(
            rtmpose_draft,
            landmarks_crop_norm=[dict(point) for point in points],
        )
        rtmpose_corrected["landmarks_crop_norm"][8]["y"] += 0.1
        rtmpose_changed = apply_label_provenance(
            [rtmpose_corrected], {"roi_1": rtmpose_draft}, True
        )[0]
        unresolved = apply_label_provenance(
            [
                {
                    "crop_id": "roi_candidate",
                    "hand_presence": {"present": False},
                    "source": "eos_negative_candidate_unassessed",
                }
            ],
            human_reviewed=False,
        )[0]
        self.assertEqual("rtmpose", rtmpose_unchanged["label_origin"])
        self.assertEqual("rtmpose_m_hand5_v1", rtmpose_unchanged["annotation_style"])
        self.assertEqual(
            "rtmpose-m_hand5_256x256_onnx", rtmpose_unchanged["teacher_model_id"]
        )
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0807",
            rtmpose_unchanged["hand_presence_teacher_model_id"],
        )
        self.assertFalse(rtmpose_unchanged["human_modified_presence"])
        self.assertEqual("rtmpose_human_corrected", rtmpose_changed["label_origin"])
        self.assertEqual("unresolved", unresolved["label_origin"])
        self.assertEqual("unlabeled_v1", unresolved["annotation_style"])
        self.assertIsNone(unresolved["teacher_model_id"])
        self.assertIsNone(unresolved["hand_presence_teacher_model_id"])
        self.assertFalse(unresolved["human_modified_presence"])

    def test_train_positive_candidate_negative_and_quality_gate_partition(self) -> None:
        cfg = load_yaml_config(Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml")
        points_norm = [
            {"id": index, "x": 0.2 + index / 100.0, "y": 0.3 + index / 100.0}
            for index in range(21)
        ]
        points_px = [
            {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
            for point in points_norm
        ]
        positive = {
            "crop_id": "positive",
            "hand_presence": {"present": True},
            "handedness": {"label": "Left", "score": 0.9},
            "landmarks_crop_norm": points_norm,
            "landmarks_crop_px": points_px,
            "mediapipe_num_hands_detected": 1,
            "palm_score": 0.9,
            "width": 256,
            "height": 256,
        }
        negative = {
            "crop_id": "candidate",
            "hand_presence": {"present": False},
            "handedness": {"label": "unknown", "score": None},
            "landmarks_crop_norm": [],
            "landmarks_crop_px": [],
            "mediapipe_num_hands_detected": 0,
            "palm_score": 0.2,
        }
        bad_positive = dict(positive, crop_id="bad", landmarks_crop_px=[dict(p) for p in points_px])
        bad_positive["landmarks_crop_px"][0]["x"] = -1.0
        published, candidates, ignored = _partition_labels(
            [positive, negative, bad_positive], "train", cfg
        )
        self.assertEqual(["positive"], [row["crop_id"] for row in published])
        self.assertEqual(["candidate"], [row["crop_id"] for row in candidates])
        self.assertEqual(["bad"], [row["crop_id"] for row in ignored])
        eval_rows, eval_candidates, _ = _partition_labels([positive, negative], "val", cfg)
        self.assertEqual(2, len(eval_rows))
        self.assertEqual([], eval_candidates)

    def test_rtmpose_train_boundary_handedness_and_presence_quality_gates(self) -> None:
        cfg = load_yaml_config(Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml")
        points_norm = [
            {"id": index, "x": 0.25, "y": 0.5} for index in range(21)
        ]
        base_points = [
            {"id": index, "x": 64.0, "y": 128.0} for index in range(21)
        ]
        one_boundary = [dict(point) for point in base_points]
        one_boundary[0]["x"] = 0.0
        base = {
            "hand_presence": {"present": True, "score": 0.9},
            "handedness": {"label": "Left", "score": 0.9},
            "landmarks_crop_norm": points_norm,
            "landmarks_crop_px": one_boundary,
            "mediapipe_num_hands_detected": 1,
            "palm_score": 0.9,
            "width": 256,
            "height": 256,
            "split": "train",
            "proposal_kind": "runtime",
            "source": "rtmpose_m_hand5_onnx",
            "capture_source_id": CAPTURE_TRAIN,
        }
        two_boundary = [dict(point) for point in one_boundary]
        two_boundary[1]["y"] = 255.0
        passed = dict(base, crop_id="one")
        rejected = dict(base, crop_id="two", landmarks_crop_px=two_boundary)
        low_handedness = dict(
            base,
            crop_id="low-handedness",
            handedness={"label": "Right", "score": 0.69},
        )
        threshold = cfg["quality"]["rtmpose_train_hand_presence_threshold"]
        presence_at_threshold = dict(
            base,
            crop_id="presence-at-threshold",
            hand_presence={"present": True, "score": threshold},
        )
        low_presence = dict(
            base,
            crop_id="low-presence",
            hand_presence={"present": False, "score": threshold - 0.0001},
        )
        positives, candidates, ignored = _partition_labels(
            [passed, rejected, low_handedness, presence_at_threshold, low_presence],
            "train",
            cfg,
        )
        self.assertEqual(
            ["one", "presence-at-threshold"],
            [row["crop_id"] for row in positives],
        )
        self.assertEqual([], candidates)
        self.assertEqual(
            ["two", "low-handedness", "low-presence"],
            [row["crop_id"] for row in ignored],
        )
        self.assertEqual("rtmpose_boundary_coordinate_gate", ignored[0]["ignore_reason"])
        self.assertEqual(
            "automatic_positive_failed_quality_gate", ignored[1]["ignore_reason"]
        )
        self.assertEqual("rtmpose_hand_presence_gate", ignored[2]["ignore_reason"])
        self.assertTrue(
            any(
                error.startswith("rtmpose_hand_presence_score_below_threshold:")
                for error in ignored[2]["quality_gate"]["errors"]
            )
        )

        eval_row = dict(low_presence, crop_id="eval", split="val")
        eval_rows, _, eval_ignored = _partition_labels([eval_row], "val", cfg)
        self.assertEqual(["eval"], [row["crop_id"] for row in eval_rows])
        self.assertEqual([], eval_ignored)
        mediapipe_row = dict(
            low_presence, crop_id="mediapipe", source="mediapipe_tasks", split="train"
        )
        mediapipe_rows, mediapipe_candidates, mediapipe_ignored = _partition_labels(
            [mediapipe_row], "train", cfg
        )
        self.assertEqual([], mediapipe_rows)
        self.assertEqual(["mediapipe"], [row["crop_id"] for row in mediapipe_candidates])
        self.assertEqual([], mediapipe_ignored)
        candidate_row = dict(
            low_presence, crop_id="candidate", proposal_kind="negative_candidate"
        )
        _, candidate_rows, candidate_ignored = _partition_labels(
            [candidate_row], "train", cfg
        )
        self.assertEqual(["candidate"], [row["crop_id"] for row in candidate_rows])
        self.assertEqual([], candidate_ignored)

    def test_rtmpose_train_presence_gate_rejects_missing_or_non_finite_scores(self) -> None:
        cfg = load_yaml_config(Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml")
        base = {
            "crop_id": "missing",
            "hand_presence": {"present": True},
            "handedness": {"label": "Left", "score": 0.9},
            "landmarks_crop_norm": [
                {"id": index, "x": 0.25, "y": 0.5} for index in range(21)
            ],
            "landmarks_crop_px": [
                {"id": index, "x": 64.0, "y": 128.0} for index in range(21)
            ],
            "width": 256,
            "height": 256,
            "split": "train",
            "proposal_kind": "runtime",
            "source": "rtmpose_m_hand5_onnx",
            "capture_source_id": CAPTURE_TRAIN,
        }
        non_finite = dict(
            base,
            crop_id="non-finite",
            hand_presence={"present": True, "score": float("nan")},
        )
        positives, candidates, ignored = _partition_labels(
            [base, non_finite], "train", cfg
        )
        self.assertEqual([], positives)
        self.assertEqual([], candidates)
        self.assertEqual(["missing", "non-finite"], [row["crop_id"] for row in ignored])
        self.assertTrue(
            all(row["ignore_reason"] == "rtmpose_hand_presence_gate" for row in ignored)
        )

    def test_rtmpose_train_connection_length_gate_toggle_scope_and_validation(self) -> None:
        cfg = load_yaml_config(Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml")
        base_points = [
            {"id": index, "x": 100.0, "y": 100.0} for index in range(21)
        ]
        base = {
            "crop_id": "base",
            "hand_presence": {"present": True, "score": 0.9},
            "handedness": {"label": "Left", "score": 0.9},
            "landmarks_crop_norm": [
                {"id": index, "x": 0.4, "y": 0.4} for index in range(21)
            ],
            "landmarks_crop_px": base_points,
            "width": 256,
            "height": 256,
            "split": "train",
            "proposal_kind": "runtime",
            "source": "rtmpose_m_hand5_onnx",
            "capture_source_id": "white-near-bright-random-train-s01-peak",
        }
        connection_threshold = float(
            cfg["quality"]["rtmpose_train_connection_length_thresholds_px"]["near"][
                "19-20"
            ]
        )
        at_threshold_points = [dict(point) for point in base_points]
        at_threshold_points[20]["x"] = 100.0 + connection_threshold
        over_threshold_points = [dict(point) for point in base_points]
        over_threshold_points[20]["x"] = 100.01 + connection_threshold
        zero_length = dict(base, crop_id="zero")
        at_threshold = dict(
            base, crop_id="at-threshold", landmarks_crop_px=at_threshold_points
        )
        over_threshold = dict(
            base, crop_id="over-threshold", landmarks_crop_px=over_threshold_points
        )
        positives, candidates, ignored = _partition_labels(
            [zero_length, at_threshold, over_threshold], "train", cfg
        )
        self.assertEqual(["zero", "at-threshold"], [row["crop_id"] for row in positives])
        self.assertEqual([], candidates)
        self.assertEqual(["over-threshold"], [row["crop_id"] for row in ignored])
        self.assertEqual("rtmpose_connection_length_gate", ignored[0]["ignore_reason"])
        self.assertTrue(
            any(
                error.startswith("rtmpose_connection_length_exceeded:19-20:")
                for error in ignored[0]["quality_gate"]["errors"]
            )
        )

        default_enabled = json.loads(json.dumps(cfg))
        default_enabled["quality"].pop("rtmpose_train_connection_length_gate_enabled")
        _, _, default_ignored = _partition_labels(
            [over_threshold], "train", default_enabled
        )
        self.assertEqual("rtmpose_connection_length_gate", default_ignored[0]["ignore_reason"])

        disabled = json.loads(json.dumps(cfg))
        disabled["quality"]["rtmpose_train_connection_length_gate_enabled"] = False
        disabled["quality"].pop("rtmpose_train_connection_length_thresholds_px")
        disabled_row = dict(
            over_threshold,
            crop_id="disabled",
            capture_source_id="not-a-valid-source-id",
        )
        disabled_positive, _, disabled_ignored = _partition_labels(
            [disabled_row], "train", disabled
        )
        self.assertEqual(["disabled"], [row["crop_id"] for row in disabled_positive])
        self.assertEqual([], disabled_ignored)

        eval_row = dict(over_threshold, crop_id="eval", split="val")
        eval_positive, _, eval_ignored = _partition_labels([eval_row], "val", cfg)
        self.assertEqual(["eval"], [row["crop_id"] for row in eval_positive])
        self.assertEqual([], eval_ignored)

        mediapipe_row = dict(
            over_threshold, crop_id="mediapipe", source="mediapipe_tasks"
        )
        mediapipe_positive, _, mediapipe_ignored = _partition_labels(
            [mediapipe_row], "train", cfg
        )
        self.assertEqual(["mediapipe"], [row["crop_id"] for row in mediapipe_positive])
        self.assertEqual([], mediapipe_ignored)

        invalid_points = [dict(point) for point in base_points]
        invalid_points[20]["id"] = 19
        invalid_row = dict(
            base, crop_id="invalid", landmarks_crop_px=invalid_points
        )
        _, _, invalid_ignored = _partition_labels([invalid_row], "train", cfg)
        self.assertEqual("rtmpose_connection_length_gate", invalid_ignored[0]["ignore_reason"])
        self.assertIn(
            "rtmpose_connection_length_landmarks_invalid",
            invalid_ignored[0]["quality_gate"]["errors"],
        )
        invalid_coordinate_points = [dict(point) for point in base_points]
        invalid_coordinate_points[20]["x"] = "bad"
        invalid_coordinate_row = dict(
            base,
            crop_id="invalid-coordinate",
            landmarks_crop_px=invalid_coordinate_points,
        )
        _, _, invalid_coordinate_ignored = _partition_labels(
            [invalid_coordinate_row], "train", cfg
        )
        self.assertEqual(
            "rtmpose_connection_length_gate",
            invalid_coordinate_ignored[0]["ignore_reason"],
        )

        bad_switch = json.loads(json.dumps(cfg))
        bad_switch["quality"]["rtmpose_train_connection_length_gate_enabled"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            _partition_labels([base], "train", bad_switch)

        missing_threshold = json.loads(json.dumps(cfg))
        missing_threshold["quality"][
            "rtmpose_train_connection_length_thresholds_px"
        ]["near"].pop("19-20")
        with self.assertRaisesRegex(ValueError, "must define exactly the 20"):
            _partition_labels([base], "train", missing_threshold)

        unknown_distance = dict(
            base,
            capture_source_id="white-unknown-bright-random-train-s01-peak",
        )
        with self.assertRaisesRegex(ValueError, "no thresholds for distance"):
            _partition_labels([unknown_distance], "train", cfg)

    def test_visualization_clean_and_variant_delete_keep_tombstone(self) -> None:
        crop, row = self._registered_roi()
        source = self._source()
        paths = proposal_paths(source, "p01")
        for key in ("palm", "reviewed", "labels", "qc"):
            paths[key].mkdir(parents=True, exist_ok=True)
            (paths[key] / "artifact.txt").write_text(key, encoding="utf-8")
        report = {
            "schema_version": "hlmf_dataset_v1",
            "dataset_id": "national-r1",
            "capture_source_id": CAPTURE_TRAIN,
            "split": "train",
            "proposal_variant": "p01",
            "raw_images": 1,
            "rois": 1,
            "published_labels": 1,
            "candidate_negatives": 0,
            "ignored": 0,
            "labels_relpath": str(
                (paths["labels"] / "hand_training_labels.jsonl").relative_to(self.root)
            ).replace("\\", "/"),
        }
        write_json(paths["qc"] / "source_publish_report.json", report)
        roi_visualization = paths["roi"] / "hand_landmarks_roi_visualization"
        roi_visualization.mkdir(parents=True)
        (roi_visualization / "preview.png").write_bytes(b"preview")
        original_root = source / "visualizations" / "original_image_landmarks"
        (original_root / "p01").mkdir(parents=True)
        (original_root / "p01" / "preview.png").write_bytes(b"preview")
        (original_root / "p01.mp4").write_bytes(b"video")
        write_json(paths["qc"] / "roi_visualization_report.json", {})
        write_json(paths["qc"] / "original_image_visualization_report.json", {})

        cleaned = clean_variant_visualizations(
            self.root, "pretrain", "national-r1", CAPTURE_TRAIN, "p01"
        )
        self.assertEqual(5, cleaned["removed_count"])
        self.assertTrue(crop.is_file())
        self.assertTrue((paths["qc"] / "source_publish_report.json").is_file())
        (original_root / "p01").mkdir(parents=True)
        (original_root / "p01" / "preview.png").write_bytes(b"preview")
        (original_root / "p01.mp4").write_bytes(b"video")

        _dataset_manifest(self.root, "pretrain", "national-r1")
        args = SimpleNamespace(
            dataset_root=str(self.root),
            scope="pretrain",
            dataset_id="national-r1",
            capture_source_id=CAPTURE_TRAIN,
            proposal_variant="p01",
            confirm_delete="wrong",
        )
        with self.assertRaisesRegex(DatasetContractError, "exactly match"):
            _run_delete_source_variant(args)
        args.confirm_delete = "p01"
        deleted = _run_delete_source_variant(args)
        self.assertEqual("retired", deleted["registry_status"])
        self.assertTrue((source / "images" / "frame001.tiff").is_file())
        self.assertTrue((source / "raw_images.jsonl").is_file())
        self.assertTrue((source / "source.json").is_file())
        for key in ("palm", "roi", "reviewed", "labels", "qc"):
            self.assertFalse(paths[key].exists())
        manifest = json.loads(
            (source.parent / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual([], manifest["capture_sources"])
        self.assertEqual(
            "retired", WarehouseRegistry(self.root).variant_status(CAPTURE_TRAIN, "p01")
        )
        self.assertEqual(0, _run_delete_source_variant(args)["removed_count"])
        with self.assertRaisesRegex(DatasetContractError, "retired"):
            WarehouseRegistry(self.root).assert_variant_writable(CAPTURE_TRAIN, "p01")
        with self.assertRaisesRegex(DatasetContractError, "retired"):
            WarehouseRegistry(self.root).register_rois([row])

    def test_negative_review_publishes_independent_copies_and_unique_registry(self) -> None:
        crop, row = self._registered_roi()
        cfg = load_yaml_config(
            Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml"
        )
        classifier = SimpleNamespace(
            model_id="hand-classifier-handedness-handpresence-0814",
            provider="CPUExecutionProvider",
            fallback_reason=None,
            classify_batch=lambda images: [
                {
                    "handedness": {"label": "Right", "score": 0.9},
                    "hand_presence": {"present": False, "score": 0.49},
                }
                for _ in images
            ],
        )
        with patch(
            "hand_autolabel.dataset_v3.HandClassifierONNX", return_value=classifier
        ) as classifier_factory:
            result = prepare_negative_review(
                self.root,
                "neg-r1",
                [row],
                cfg,
                Path(__file__).resolve().parents[1],
            )
        classifier_factory.assert_called_once_with(
            Path(__file__).resolve().parents[1]
            / "models/hand_classifier/handedness-handpresence-0814/model.onnx",
            "auto",
        )
        review_image = next(Path(result["review_root"]).glob("images/*/*"))
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0814",
            result["hand_classifier_model_id"],
        )
        self.assertNotEqual(os.stat(crop).st_ino, os.stat(review_image).st_ino)
        self.assertEqual(crop.read_bytes(), review_image.read_bytes())
        manifest = publish_negative_review(self.root, "neg-r1")
        published = next(
            (self.root / "GoldSource" / "NegativeSamples" / "neg-r1" / "published" / "images").glob("*/*")
        )
        self.assertNotEqual(os.stat(crop).st_ino, os.stat(published).st_ino)
        self.assertEqual(crop.read_bytes(), published.read_bytes())
        self.assertEqual("copied_review_and_published_images", manifest["image_policy"])
        self.assertEqual(manifest["records"], 1)
        delete_source_variant(
            self.root,
            "pretrain",
            "national-r1",
            CAPTURE_TRAIN,
            "p01",
            "p01",
        )
        self.assertFalse(crop.exists())
        self.assertTrue(published.is_file())
        self.assertGreater(published.stat().st_size, 0)
        with patch(
            "hand_autolabel.dataset_v3.HandClassifierONNX",
            return_value=classifier,
        ), self.assertRaises(DatasetContractError):
            prepare_negative_review(
                self.root,
                "neg-r1",
                [row],
                cfg,
                Path(__file__).resolve().parents[1],
            )

    def test_negative_review_precheck_uses_strict_threshold_and_writes_excluded_manifest(self) -> None:
        crop, row = self._registered_roi()
        second_crop = crop.with_name("crop2.png")
        cv2.imwrite(str(second_crop), np.ones((256, 256), dtype=np.uint8))
        second = dict(
            row,
            roi_id="roi_test002",
            proposal_slot=1,
            crop_relpath=str(second_crop.relative_to(self.root)).replace("\\", "/"),
        )
        WarehouseRegistry(self.root).register_rois([second])
        cfg = load_yaml_config(
            Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml"
        )
        classifier = SimpleNamespace(
            model_id="hand-classifier-handedness-handpresence-0809",
            provider="CUDAExecutionProvider",
            fallback_reason=None,
            classify_batch=lambda _images: [
                {
                    "handedness": {"label": "Right", "score": 0.9},
                    "hand_presence": {"present": False, "score": 0.499},
                },
                {
                    "handedness": {"label": "Right", "score": 0.9},
                    "hand_presence": {"present": True, "score": 0.5},
                },
            ],
        )
        with patch(
            "hand_autolabel.dataset_v3.HandClassifierONNX",
            return_value=classifier,
        ):
            result = prepare_negative_review(
                self.root,
                "neg-threshold",
                [row, second],
                cfg,
                Path(__file__).resolve().parents[1],
            )
        review_root = Path(result["review_root"])
        selected = read_jsonl(review_root / "candidate_manifest.jsonl")
        excluded = read_jsonl(review_root / "precheck_excluded.jsonl")
        self.assertEqual(1, result["candidate_count"])
        self.assertEqual(1, result["precheck_excluded_count"])
        self.assertEqual("roi_test001", selected[0]["roi_id"])
        self.assertEqual("roi_test002", excluded[0]["roi_id"])
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809",
            selected[0]["negative_review_precheck"]["model_id"],
        )
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809",
            excluded[0]["negative_review_precheck"]["model_id"],
        )
        self.assertFalse(
            excluded[0]["negative_review_precheck"]["selected_for_human_review"]
        )

    def test_quality_gate_rejection_statistics_are_exclusive_and_aggregated(self) -> None:
        ignored = [
            {"ignore_reason": "rtmpose_hand_presence_gate", "quality_gate": {}},
            {"ignore_reason": "rtmpose_boundary_coordinate_gate", "quality_gate": {}},
            {"ignore_reason": "rtmpose_connection_length_gate", "quality_gate": {}},
            {
                "ignore_reason": "automatic_positive_failed_quality_gate",
                "quality_gate": {"warnings": ["low_handedness_score:0.600"]},
            },
            {
                "ignore_reason": "automatic_positive_failed_quality_gate",
                "quality_gate": {"warnings": ["multiple_hands_in_one_crop"]},
            },
        ]
        counts = _quality_gate_rejection_counts(ignored)
        self.assertEqual(
            {
                "hand_presence": 1,
                "boundary_coordinate": 1,
                "connection_length": 1,
                "handedness": 1,
            },
            counts,
        )
        source, _ = self._validated_source()
        report_dir = source / "qc" / "p01"
        report_dir.mkdir(parents=True)
        write_json(
            report_dir / "source_publish_report.json",
            {
                "capture_source_id": CAPTURE_TRAIN,
                "proposal_variant": "p01",
                "rois": 5,
                "quality_gate_rejections": counts,
            },
        )
        manifest = _dataset_manifest(self.root, "pretrain", "national-r1")
        self.assertEqual(counts, manifest["quality_gate_rejections"])
        self.assertEqual(
            counts,
            manifest["quality_gate_rejections_by_capture_source_id"][CAPTURE_TRAIN],
        )

    def test_selection_review_publishes_independent_images(self) -> None:
        crop, row = self._registered_roi()
        row = dict(row, hand_presence={"present": True})
        result = prepare_selection_review(self.root, "hard-r1", [row])
        self.assertTrue(Path(result["review_root"]).is_dir())
        manifest = publish_selection_review(self.root, "hard-r1")
        published = self.root / "Selections" / "hard-r1" / "published"
        self.assertEqual(manifest["image_policy"], "copied_review_and_published_images")
        published_image = next((published / "images").glob("*/*"))
        self.assertNotEqual(os.stat(crop).st_ino, os.stat(published_image).st_ino)
        selected = read_jsonl(published / "selection.jsonl")[0]
        self.assertEqual(selected["roi_id"], row["roi_id"])
        self.assertTrue((self.root / selected["published_relpath"]).is_file())
        delete_source_variant(
            self.root,
            "pretrain",
            "national-r1",
            CAPTURE_TRAIN,
            "p01",
            "p01",
        )
        self.assertFalse(crop.exists())
        self.assertTrue(published_image.is_file())
        self.assertGreater(published_image.stat().st_size, 0)

    def test_hard_review_uses_cvat_and_publishes_under_gold_source(self) -> None:
        crop, row = self._registered_roi()
        points_px = [
            {"id": index, "x": 24.0 + index * 4.0, "y": 32.0 + index * 3.0}
            for index in range(21)
        ]
        request = dict(
            row,
            proposal_kind="runtime",
            hand_presence={"present": True},
            handedness={"label": "Left", "score": 0.9},
            landmarks_crop_px=points_px,
            landmarks_crop_norm=[
                {"id": point["id"], "x": point["x"] / 255.0, "y": point["y"] / 255.0}
                for point in points_px
            ],
            landmarks_image_px=points_px,
            roi_corners_px=[[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]],
            source="rtmpose_m_hand5_onnx",
        )
        args = _parser().parse_args(
            ["registry-check", "--dataset-root", str(self.root)]
        )
        cfg = _load_public_configs(args)
        prepared = prepare_hard_review(self.root, "hard-gold-r1", [request], cfg)
        review = Path(prepared["review_root"])
        xml_text = (review / "cvat_autolabel.xml").read_text(encoding="utf-8")
        self.assertIn("<version>1.1</version>", xml_text)
        self.assertIn('label="hand_landmarks"', xml_text)
        (review / "cvat_reviewed.xml").write_text(xml_text, encoding="utf-8")
        imported = import_hard_review(self.root, "hard-gold-r1", cfg)
        self.assertEqual(1, imported["reviewed_rows"])
        manifest = publish_hard_review(self.root, "hard-gold-r1")
        self.assertEqual(1, manifest["positive"])
        self.assertEqual("cvat_xml_1.1_precise_hand_roi_review", manifest["review_contract"])
        published = (
            self.root / "GoldSource" / "HardSamples" / "hard-gold-r1" / "published"
        )
        label = read_jsonl(published / "hard_labels.jsonl")[0]
        self.assertTrue(label["human_reviewed"])
        self.assertTrue((self.root / label["published_relpath"]).is_file())
        crop.unlink()
        self.assertTrue((self.root / label["published_relpath"]).is_file())

    def test_local_downsample_is_tiff_only_and_refuses_overwrite(self) -> None:
        source = self.root / "camera"
        for index in range(5):
            write_tiff(source / f"{index:03d}.tiff", shape=(16, 16), value=index)
        output = self.root / "retained"
        report = downsample(source, 2, output)
        self.assertEqual(report["retained_frames"], 3)
        with self.assertRaises(FileExistsError):
            downsample(source, 2, output)

    def test_progress_wrapper_enables_tqdm_with_known_total(self) -> None:
        with patch("hand_autolabel.progress.tqdm", return_value=iter([1, 2, 3])) as mocked:
            self.assertEqual(
                [1, 2, 3],
                list(
                    track_progress(
                        [1, 2, 3],
                        enabled=True,
                        description="test stage",
                        unit="item",
                    )
                ),
            )
        self.assertEqual(3, mocked.call_args.kwargs["total"])
        self.assertEqual("test stage", mocked.call_args.kwargs["desc"])

    def test_roi_visualization_samples_train_and_renders_all_eval(self) -> None:
        roi_images = self.root / "roi_images"
        roi_images.mkdir(parents=True)
        rows = []
        for index in range(10):
            crop_name = f"roi_{index:02d}.png"
            image = np.zeros((256, 256), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(roi_images / crop_name), image))
            points = [
                {"id": landmark_id, "x": 20 + landmark_id * 8, "y": 30 + landmark_id * 5}
                for landmark_id in range(21)
            ]
            rows.append(
                {
                    "crop_path": crop_name,
                    "hand_presence": {"present": True},
                    "handedness": {"label": "Right", "score": 0.9},
                    "landmarks_crop_px": points,
                }
            )

        sampled = evenly_spaced_sample(rows, 4)
        self.assertEqual(
            ["roi_00.png", "roi_03.png", "roi_06.png", "roi_09.png"],
            [row["crop_path"] for row in sampled],
        )

        train_output = self.root / "train_visualization"
        train_output.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(train_output / "stale.png"), np.zeros((8, 8), dtype=np.uint8)))
        train_stats = render_autolabel_roi_visualizations(
            rows,
            roi_images,
            train_output,
            split="train",
            train_max_samples=4,
        )
        self.assertEqual("evenly_spaced", train_stats["selection"])
        self.assertEqual(4, train_stats["saved"])
        self.assertEqual(1, train_stats["stale_removed"])
        self.assertEqual(
            {"roi_00.png", "roi_03.png", "roi_06.png", "roi_09.png"},
            {path.name for path in train_output.glob("*.png")},
        )
        rendered = cv2.imread(str(train_output / "roi_00.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered.max()), 0)

        eval_output = self.root / "eval_visualization"
        eval_stats = render_autolabel_roi_visualizations(
            rows,
            roi_images,
            eval_output,
            split="val",
            train_max_samples=4,
        )
        self.assertEqual("all", eval_stats["selection"])
        self.assertEqual(10, eval_stats["saved"])
        self.assertEqual(10, len(list(eval_output.glob("*.png"))))

    def test_cli_visualization_override_has_priority_over_config(self) -> None:
        args = _parser().parse_args(
            [
                "autolabel-train",
                "--dataset-root",
                str(self.root),
                "--scope",
                "pretrain",
                "--dataset-id",
                "national-r1",
                "--capture-source-id",
                CAPTURE_TRAIN,
                "--proposal-variant",
                "p01",
                "--roi-visualization",
                "false",
                "--original-visualization",
                "false",
            ]
        )
        with patch.dict(
            os.environ,
            {
                "AUTOLABEL_OVERRIDES": (
                    '{"visualization":{"roi_enabled":true,'
                    '"original_image_enabled":true}}'
                )
            },
        ):
            cfg = _load_public_configs(args)
        self.assertFalse(cfg["visualization"]["roi_enabled"])
        self.assertFalse(cfg["visualization"]["original_image_enabled"])

    def test_hand_landmark_backend_global_default_and_cli_override(self) -> None:
        common = [
            "autolabel-train",
            "--dataset-root",
            str(self.root),
            "--scope",
            "pretrain",
            "--dataset-id",
            "national-r1",
            "--capture-source-id",
            CAPTURE_TRAIN,
            "--proposal-variant",
            "p01",
        ]
        global_cfg = _load_public_configs(_parser().parse_args(common))
        self.assertEqual("rtmpose_onnx", global_cfg["hand_landmark"]["backend"])
        override_cfg = _load_public_configs(
            _parser().parse_args(common + ["--hand-landmark-backend", "mediapipe_tasks"])
        )
        self.assertEqual("mediapipe_tasks", override_cfg["hand_landmark"]["backend"])

    def test_eval_limit_preflight_uses_pending_report_without_writing_manifest(self) -> None:
        dataset = self.root / "EValSource" / "eval-r1"
        existing = dataset / "existing-test"
        pending = dataset / "pending-test"
        for source, capture_id, raw_count in (
            (existing, "existing-test", 2000),
            (pending, "pending-test", 600),
        ):
            source.mkdir(parents=True)
            (source / "source.json").write_text(
                json.dumps(
                    {
                        "capture_source_id": capture_id,
                        "split": "test",
                        "raw_image_count": raw_count,
                    }
                ),
                encoding="utf-8",
            )
        report_dir = existing / "qc" / "eos-1.0"
        report_dir.mkdir(parents=True)
        (report_dir / "source_publish_report.json").write_text(
            json.dumps(
                {
                    "capture_source_id": "existing-test",
                    "proposal_variant": "eos-1.0",
                    "rois": 1800,
                }
            ),
            encoding="utf-8",
        )
        prospective = _dataset_manifest(
            self.root,
            "eval",
            "eval-r1",
            pending_report={
                "capture_source_id": "pending-test",
                "proposal_variant": "eos-1.0",
                "rois": 500,
            },
            write=False,
        )
        self.assertFalse((dataset / "dataset_manifest.json").exists())
        cfg = {
            "evaluation_limits": {
                "max_raw_images_per_split": 2500,
                "max_rois_per_split": 3000,
            }
        }
        with self.assertRaisesRegex(
            DatasetContractError, "test exceeds raw-image limit: 2600"
        ):
            _validate_evaluation_limits(prospective, cfg)

        cfg["evaluation_limits"]["max_raw_images_per_split"] = 2600
        _validate_evaluation_limits(prospective, cfg)

    def test_dataset_manifest_excludes_unpublished_capture_source(self) -> None:
        source = source_root(self.root, "eval", "eval-r1", CAPTURE_VAL)
        source.mkdir(parents=True)
        write_json(
            source / "source.json",
            {
                "schema_version": "hlmf_dataset_v1",
                "scope": "eval",
                "dataset_id": "eval-r1",
                "capture_source_id": CAPTURE_VAL,
                "split": "val",
                "raw_image_count": 1,
            },
        )

        unpublished = _dataset_manifest(self.root, "eval", "eval-r1")
        self.assertEqual([], unpublished["capture_sources"])

        report_dir = source / "qc" / "p01"
        report_dir.mkdir(parents=True)
        write_json(
            report_dir / "source_publish_report.json",
            {
                "capture_source_id": CAPTURE_VAL,
                "proposal_variant": "p01",
                "rois": 1,
            },
        )
        published = _dataset_manifest(self.root, "eval", "eval-r1")
        self.assertEqual(
            [CAPTURE_VAL],
            [row["capture_source_id"] for row in published["capture_sources"]],
        )

        with self.assertRaisesRegex(DatasetContractError, "dataset_id must use"):
            _dataset_manifest(self.root, "eval", "../outside")
        self.assertFalse((self.root / "outside" / "dataset_manifest.json").exists())

    def test_standalone_roi_visualization_reuses_existing_draft(self) -> None:
        source = source_root(self.root, "pretrain", "national-r1", CAPTURE_TRAIN)
        paths = proposal_paths(source, "p01")
        crop_dir = paths["roi"] / "images"
        crop_dir.mkdir(parents=True)
        rows = []
        for index in range(3):
            crop_name = f"standalone_{index}.png"
            self.assertTrue(
                cv2.imwrite(str(crop_dir / crop_name), np.zeros((256, 256), dtype=np.uint8))
            )
            rows.append(
                {
                    "crop_path": f"ignored/parent/{crop_name}",
                    "hand_presence": {"present": False},
                    "handedness": {"label": "unknown", "score": None},
                    "landmarks_crop_px": [],
                    "proposal_kind": "runtime" if index == 0 else "negative_candidate",
                    "source": (
                        "rtmpose_m_hand5_onnx"
                        if index == 0
                        else "eos_negative_candidate_unassessed"
                    ),
                }
            )
        write_jsonl(paths["roi"] / "hand_landmarks_autolabel_draft.jsonl", rows)
        write_json(
            paths["qc"] / "mediapipe_report.json",
            {"hand_landmark_backend": "rtmpose_onnx"},
        )

        args = SimpleNamespace(
            dataset_root=str(self.root),
            scope="pretrain",
            dataset_id="national-r1",
            capture_source_id=CAPTURE_TRAIN,
            proposal_variant="p01",
        )
        cfg = {
            "dataset": {},
            "paths": {},
            "palm": {"keep_low_score_candidates_for_negatives": True},
            "visualization": {"roi_enabled": False, "train_max_samples": 2},
        }
        report = _run_existing_roi_visualization(
            args,
            cfg,
            show_progress=False,
        )
        self.assertTrue(report["enabled"])
        self.assertEqual("standalone", report["trigger"])
        self.assertEqual("evenly_spaced", report["selection"])
        self.assertEqual(3, report["input_rows"])
        self.assertEqual(2, report["excluded_non_runtime"])
        self.assertEqual(1, report["saved"])
        self.assertEqual(
            1,
            len(
                list(
                    (paths["roi"] / "hand_landmarks_roi_visualization").glob("*.png")
                )
            ),
        )
        self.assertTrue((paths["qc"] / "roi_visualization_report.json").is_file())
        self.assertFalse((paths["qc"] / "autolabel_visualization_report.json").exists())

    def test_original_image_visualization_preserves_source_stems_as_png(self) -> None:
        source_images = self.root / "source_images"
        write_tiff(source_images / "frame_alpha.tif", shape=(96, 128), value=31)
        write_tiff(source_images / "frame_beta.tiff", shape=(96, 128), value=47)
        points = [
            {
                "id": landmark_id,
                "x": 15 + (landmark_id % 5) * 18,
                "y": 25 + (landmark_id // 5) * 13,
            }
            for landmark_id in range(21)
        ]
        rows = [
            {
                "image": "frame_alpha.tif",
                "hand_presence": {"present": True},
                "handedness": {"label": "Right", "score": 0.91},
                "landmarks_image_px": points,
            },
            {
                "image": "frame_beta.tiff",
                "hand_presence": {"present": False},
                "handedness": {"label": "unknown", "score": None},
                "landmarks_image_px": [],
            },
        ]
        output = self.root / "original_visualization" / "p01"
        write_tiff(output / "stale.tif", shape=(8, 8), value=0)
        stats = render_original_image_visualizations(
            rows,
            source_images,
            output,
            proposal_variant="p01",
        )

        expected_names = {"frame_alpha.png", "frame_beta.png"}
        self.assertEqual(expected_names, {path.name for path in output.iterdir()})
        self.assertEqual(2, stats["saved"])
        self.assertEqual(1, stats["images_with_hands"])
        self.assertEqual(1, stats["images_without_hands"])
        self.assertEqual(1, stats["positive_hands"])
        self.assertEqual(1, stats["teacher_abstain_rois"])
        self.assertEqual(1, stats["stale_removed"])
        self.assertEqual("png", stats["output_format"])
        self.assertEqual(3, stats["png_compression"])
        rendered = cv2.imread(str(output / "frame_alpha.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(rendered)
        self.assertEqual(3, rendered.ndim)
        self.assertGreater(int(rendered.max()), 31)

        second_output = self.root / "original_visualization" / "p02"
        render_original_image_visualizations(
            rows,
            source_images,
            second_output,
            proposal_variant="p02",
        )
        self.assertEqual(expected_names, {path.name for path in second_output.iterdir()})

        collision_images = self.root / "collision_images"
        write_tiff(collision_images / "duplicate.tif", shape=(8, 8), value=0)
        write_tiff(collision_images / "duplicate.tiff", shape=(8, 8), value=0)
        with self.assertRaisesRegex(
            TrainingRoiVisualizationError,
            "stems collide after PNG conversion",
        ):
            render_original_image_visualizations(
                [],
                collision_images,
                self.root / "collision_output",
                proposal_variant="p01",
            )

    def test_standalone_original_visualization_reuses_existing_draft(self) -> None:
        source = source_root(self.root, "eval", "national-r1", CAPTURE_VAL)
        paths = proposal_paths(source, "p01")
        write_tiff(source / "images" / "original_001.tiff", shape=(96, 128), value=21)
        points = [
            {"id": landmark_id, "x": 20 + landmark_id, "y": 30 + landmark_id}
            for landmark_id in range(21)
        ]
        write_jsonl(
            paths["roi"] / "hand_landmarks_autolabel_draft.jsonl",
            [
                {
                    "image": "original_001.tiff",
                    "hand_presence": {"present": True},
                    "handedness": {"label": "Left", "score": 0.8},
                    "landmarks_image_px": points,
                }
            ],
        )
        args = SimpleNamespace(
            dataset_root=str(self.root),
            scope="eval",
            dataset_id="national-r1",
            capture_source_id=CAPTURE_VAL,
            proposal_variant="p01",
        )
        report = _run_existing_original_image_visualization(
            args,
            {"visualization": {"original_video_enabled": False}},
            show_progress=False,
        )

        output = source / "visualizations" / "original_image_landmarks" / "p01"
        self.assertTrue(report["enabled"])
        self.assertEqual("standalone", report["trigger"])
        self.assertEqual(1, report["saved"])
        self.assertEqual({"original_001.png"}, {path.name for path in output.iterdir()})
        self.assertTrue(
            (paths["qc"] / "original_image_visualization_report.json").is_file()
        )

    def test_public_makefile_has_no_palm_review_or_manual_roi_interface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        makefile = (root / "Makefile").read_text(encoding="utf-8")
        self.assertNotIn("palm-cvat", makefile)
        self.assertNotIn("import_palm", makefile)
        self.assertIn("Hand ROIs are always program-generated", makefile)
        self.assertIn("autolabel-visualize-roi:", makefile)
        self.assertNotIn("\nautolabel-visualize:", makefile)
        self.assertNotIn("\nVISUALIZATION ?=", makefile)
        self.assertIn("autolabel-visualize-original:", makefile)
        self.assertIn("autolabel-visualizations-clean:", makefile)
        self.assertIn("source-variant-delete:", makefile)
        self.assertIn("batch-eval-autolabel:", makefile)
        self.assertIn("batch-train-autolabel:", makefile)
        self.assertIn("batch-autolabel-visualizations-clean:", makefile)
        self.assertIn("batch-source-variant-delete:", makefile)
        self.assertIn("palm-distance-check:", makefile)
        self.assertIn("dataset-manifest-rebuild:", makefile)
        self.assertEqual({"hlmf.py"}, {path.name for path in (root / "scripts").glob("*.py")})
        self.assertEqual(
            {
                "batch_autolabel_visualizations_clean.sh",
                "batch_eval_autolabel.sh",
                "batch_source_variant_delete.sh",
                "batch_train_autolabel.sh",
            },
            {path.name for path in (root / "scripts").glob("*.sh")},
        )
        batch_eval = (root / "scripts" / "batch_eval_autolabel.sh").read_text(encoding="utf-8")
        self.assertIn('[[ -d "$source_dir/images" ]]', batch_eval)
        self.assertNotIn("-name source.json", batch_eval)
        for script_name in ("batch_eval_autolabel.sh", "batch_train_autolabel.sh"):
            batch_script = (root / "scripts" / script_name).read_text(encoding="utf-8")
            self.assertIn("check-palm-distance", batch_script)
            self.assertIn("SKIPPED_UNSUPPORTED_DISTANCE", batch_script)
            self.assertIn("SUPPORTED_SOURCE_DIRS", batch_script)
            self.assertIn("Skipped source IDs", batch_script)
        self.assertEqual(
            {"autolabel.yaml", "review.yaml", "datasets.yaml", "cvat_label.json"},
            {path.name for path in (root / "configs").iterdir() if path.is_file()},
        )

        autolabel = load_yaml_config(root / "configs" / "autolabel.yaml")
        review = load_yaml_config(root / "configs" / "review.yaml")
        datasets = load_yaml_config(root / "configs" / "datasets.yaml")
        self.assertIn("palm", autolabel)
        self.assertEqual(
            {
                "roi_enabled": False,
                "original_image_enabled": False,
                "original_video_enabled": True,
                "train_max_samples": 200,
            },
            autolabel["visualization"],
        )
        self.assertNotIn("cvat", autolabel)
        self.assertIn("cvat", review)
        self.assertNotIn("palm", review)
        self.assertIn("evaluation_limits", datasets)

    def test_cvat_label_contract_keeps_hand_tags_and_21_point_skeleton(self) -> None:
        root = Path(__file__).resolve().parents[1]
        labels = json.loads((root / "configs" / "cvat_label.json").read_text(encoding="utf-8"))
        by_name = {label["name"]: label for label in labels}
        self.assertEqual(
            {
                "no_hand",
                "Left",
                "Right",
                "unknown_handedness",
                "ignore_for_training",
                "hand_landmarks",
            },
            set(by_name),
        )
        self.assertEqual("skeleton", by_name["hand_landmarks"]["type"])
        self.assertEqual(
            [str(index) for index in range(1, 22)],
            [point["name"] for point in by_name["hand_landmarks"]["sublabels"]],
        )
        review = load_yaml_config(root / "configs" / "review.yaml")["cvat"]
        self.assertEqual("configs/cvat_label.json", review["label_schema_path"])
        self.assertEqual("hand_landmarks", review["label_name"])
        self.assertEqual("no_hand", review["no_hand_label_name"])
        self.assertEqual("Left", review["left_label_name"])
        self.assertEqual("Right", review["right_label_name"])
        self.assertEqual("unknown_handedness", review["unknown_handedness_label_name"])
        self.assertEqual("ignore_for_training", review["ignore_for_training_label_name"])


if __name__ == "__main__":
    unittest.main()
