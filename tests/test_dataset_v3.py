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
    enrich_palm_rows,
    enrich_roi_rows,
    parse_capture_source_id,
    prepare_negative_review,
    prepare_selection_review,
    proposal_paths,
    publish_negative_review,
    publish_selection_review,
    source_root,
    validate_and_normalize_source,
)
from hand_autolabel.formats import load_yaml_config, read_jsonl, write_jsonl
from hand_autolabel.mediapipe_roi_visualization import (
    evenly_spaced_sample,
    render_autolabel_visualizations,
    render_original_image_visualizations,
)
from hand_autolabel.progress import track_progress
from scripts.hlmf import (
    _load_public_configs,
    _parser,
    _partition_labels,
    _run_existing_autolabel_visualization,
    _run_existing_original_image_visualization,
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

    def test_negative_review_publishes_hardlinks_and_unique_registry(self) -> None:
        crop, row = self._registered_roi()
        result = prepare_negative_review(self.root, "neg-r1", [row])
        review_image = next(Path(result["review_root"]).glob("images/*/*"))
        self.assertEqual(os.stat(crop).st_ino, os.stat(review_image).st_ino)
        manifest = publish_negative_review(self.root, "neg-r1")
        published = next(
            (self.root / "GoldSource" / "NegativeSamples" / "neg-r1" / "published" / "images").glob("*/*")
        )
        self.assertEqual(os.stat(crop).st_ino, os.stat(published).st_ino)
        self.assertEqual(manifest["records"], 1)
        with self.assertRaises(DatasetContractError):
            prepare_negative_review(self.root, "neg-r1", [row])

    def test_selection_review_is_zero_copy_after_publish(self) -> None:
        _, row = self._registered_roi()
        row = dict(row, hand_presence={"present": True})
        result = prepare_selection_review(self.root, "hard-r1", [row])
        self.assertTrue(Path(result["review_root"]).is_dir())
        manifest = publish_selection_review(self.root, "hard-r1")
        published = self.root / "Selections" / "hard-r1" / "published"
        self.assertEqual(manifest["image_policy"], "zero_copy_reference_pretrain_roi")
        self.assertFalse((published / "images").exists())
        self.assertEqual(read_jsonl(published / "selection.jsonl")[0]["roi_id"], row["roi_id"])

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

    def test_autolabel_visualization_samples_train_and_renders_all_eval(self) -> None:
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
        train_stats = render_autolabel_visualizations(
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
        eval_stats = render_autolabel_visualizations(
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
                "--visualization",
                "false",
                "--original-visualization",
                "false",
            ]
        )
        with patch.dict(
            os.environ,
            {
                "AUTOLABEL_OVERRIDES": (
                    '{"visualization":{"enabled":true,'
                    '"original_image_enabled":true}}'
                )
            },
        ):
            cfg = _load_public_configs(args)
        self.assertFalse(cfg["visualization"]["enabled"])
        self.assertFalse(cfg["visualization"]["original_image_enabled"])

    def test_standalone_visualization_reuses_existing_autolabel_draft(self) -> None:
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
                }
            )
        write_jsonl(paths["roi"] / "hand_landmarks_autolabel_draft.jsonl", rows)

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
            "visualization": {"enabled": False, "train_max_samples": 2},
        }
        report = _run_existing_autolabel_visualization(
            args,
            cfg,
            show_progress=False,
        )
        self.assertTrue(report["enabled"])
        self.assertEqual("standalone", report["trigger"])
        self.assertEqual("evenly_spaced", report["selection"])
        self.assertEqual(2, report["saved"])
        self.assertEqual(
            2,
            len(list((paths["roi"] / "hand_landmarks_visualization").glob("*.png"))),
        )

    def test_original_image_visualization_preserves_every_source_filename(self) -> None:
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

        expected_names = {"frame_alpha.tif", "frame_beta.tiff"}
        self.assertEqual(expected_names, {path.name for path in output.iterdir()})
        self.assertEqual(2, stats["saved"])
        self.assertEqual(1, stats["images_with_hands"])
        self.assertEqual(1, stats["images_without_hands"])
        self.assertEqual(1, stats["positive_hands"])
        self.assertEqual(1, stats["teacher_abstain_rois"])
        self.assertEqual(1, stats["stale_removed"])
        rendered = cv2.imread(str(output / "frame_alpha.tif"), cv2.IMREAD_UNCHANGED)
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
            show_progress=False,
        )

        output = source / "visualizations" / "original_image_landmarks" / "p01"
        self.assertTrue(report["enabled"])
        self.assertEqual("standalone", report["trigger"])
        self.assertEqual(1, report["saved"])
        self.assertEqual({"original_001.tiff"}, {path.name for path in output.iterdir()})
        self.assertTrue(
            (paths["qc"] / "original_image_visualization_report.json").is_file()
        )

    def test_public_makefile_has_no_palm_review_or_manual_roi_interface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        makefile = (root / "Makefile").read_text(encoding="utf-8")
        self.assertNotIn("palm-cvat", makefile)
        self.assertNotIn("import_palm", makefile)
        self.assertIn("Hand ROIs are always program-generated", makefile)
        self.assertIn("autolabel-visualize:", makefile)
        self.assertIn("autolabel-visualize-original:", makefile)
        self.assertEqual({"hlmf.py"}, {path.name for path in (root / "scripts").glob("*.py")})
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
                "enabled": False,
                "original_image_enabled": False,
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
