from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from hand_autolabel.cvat_io import export_cvat_xml, import_cvat_xml
from hand_autolabel.finalization import atomic_write_jsonl, finalize_training, sha256_file
from hand_autolabel.finetune_gold import (
    GoldPipelineError,
    _assert_regular_file,
    _load_dragon_image,
    _materialize_rows_from_selection,
    _safe_relative_file,
    build_pretrain_source_registry,
    export_finetune_gold,
    finalize_gold_aggregate,
    import_finetune_gold,
    import_all_finetune_gold,
    match_dragon_hands_to_palms,
    parse_dragon_hand_annotations,
    parse_dragon_palm_annotations,
)
from hand_autolabel.image_io import write_image


def _landmarks() -> list[dict]:
    return [
        {"id": index, "x": 0.30 + (index % 5) * 0.05, "y": 0.25 + (index // 5) * 0.08}
        for index in range(21)
    ]


class DragonContractTests(unittest.TestCase):
    def test_parsers_matching_and_exif_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            points = " ".join(["0.2", "0.3"] * 21)
            (root / "hand.txt").write_text(f"a.jpg 1 {points}\n", encoding="utf-8")
            (root / "palm.txt").write_text(
                "frame: a.jpg 0.1 0.2 0.4 0.5 0.2 0.45 0.25 0.30\n", encoding="utf-8"
            )
            hands = parse_dragon_hand_annotations(root / "hand.txt")
            palms = parse_dragon_palm_annotations(root / "palm.txt")
            self.assertEqual(match_dragon_hands_to_palms(hands["a.jpg"], palms["a.jpg"]), [0])
            self.assertIsNone(match_dragon_hands_to_palms(hands["a.jpg"], palms["a.jpg"] * 2))

            pixels = np.full((1280, 720, 3), 80, dtype=np.uint8)
            image = Image.fromarray(pixels, mode="RGB")
            exif = image.getexif()
            exif[274] = 6
            image.save(root / "a.jpg", exif=exif)
            logical, orientation = _load_dragon_image(
                root / "a.jpg", expected_orientation=6, logical_size=(1280, 720)
            )
            self.assertEqual(logical.shape, (720, 1280))
            self.assertEqual(orientation, 6)


class FinetuneGoldIntegrationTests(unittest.TestCase):
    def test_controlled_paths_reject_traversal_and_ancestor_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "package"
            package.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")

            with self.assertRaises(GoldPipelineError):
                _safe_relative_file(package, "../outside.txt", "test artifact")
            with self.assertRaises(GoldPipelineError):
                _safe_relative_file(package, outside, "test artifact")

            real = package / "real"
            real.mkdir()
            nested = real / "nested.txt"
            nested.write_text("nested", encoding="utf-8")
            linked = package / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable on this platform")
            with self.assertRaises(GoldPipelineError):
                _safe_relative_file(package, "linked/nested.txt", "test artifact")
            with self.assertRaises(GoldPipelineError):
                _assert_regular_file(linked / "nested.txt", "selection input")

    def test_selection_materializer_hashes_shared_large_artifacts_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "parent"
            image_dir = parent / "02_roi_crops" / "images"
            corners = [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]]
            manifests = []
            drafts = []
            requests = []
            for index in range(2):
                crop_id = "parent:crop{}".format(index)
                image = image_dir / "crop{}.png".format(index)
                self.assertTrue(
                    write_image(image, np.full((256, 256), 100 + index, dtype=np.uint8))
                )
                manifest = {
                    "crop_id": crop_id,
                    "image": "frame{}.tiff".format(index),
                    "palm_det_id": "parent:palm{}".format(index),
                    "palm_valid": True,
                    "palm_score": 0.9,
                    "crop_path": str(image),
                    "roi_rect": {
                        "x_center": 127.5,
                        "y_center": 127.5,
                        "width": 255.0,
                        "height": 255.0,
                        "rotation_rad": 0.0,
                    },
                    "roi_corners_px": corners,
                    "output_size": [256, 256],
                }
                manifests.append(manifest)
                drafts.append(
                    {
                        **manifest,
                        "hand_id": None,
                        "hand_presence": {"present": False},
                        "handedness": {"label": "unknown", "score": None},
                        "landmarks_crop_norm": [],
                        "landmarks_crop_px": [],
                        "landmarks_image_px": [],
                        "width": 256,
                        "height": 256,
                        "source_image_width": 1280,
                        "source_image_height": 720,
                    }
                )
            manifest_path = parent / "02_roi_crops" / "hand_roi_crops_manifest.jsonl"
            draft_path = parent / "02_roi_crops" / "hand_landmarks_autolabel_draft.jsonl"
            atomic_write_jsonl(manifest_path, manifests)
            atomic_write_jsonl(draft_path, drafts)
            manifest_sha = sha256_file(manifest_path)
            draft_sha = sha256_file(draft_path)
            for manifest in manifests:
                image = Path(manifest["crop_path"])
                requests.append(
                    {
                        "source_kind": "reviewed_hard_gold",
                        "parent_dataset_id": "parent_ds",
                        "parent_source_crop_id": manifest["crop_id"],
                        "parent_global_crop_id": "parent_ds:{}".format(manifest["crop_id"]),
                        "parent_manifest_path": str(manifest_path),
                        "parent_manifest_sha256": manifest_sha,
                        "parent_draft_path": str(draft_path),
                        "parent_draft_sha256": draft_sha,
                        "parent_crop_path": str(image),
                        "image_sha256": sha256_file(image),
                    }
                )
            request_path = root / "selection_request.jsonl"
            atomic_write_jsonl(request_path, requests)
            real_sha256 = sha256_file
            with mock.patch(
                "hand_autolabel.finetune_gold.sha256_file",
                wraps=real_sha256,
            ) as hashed:
                _materialize_rows_from_selection(
                    request_path,
                    "hard_gold",
                    "hard_gold",
                    root / "task",
                )
            hashed_paths = [Path(call.args[0]).resolve() for call in hashed.call_args_list]
            self.assertEqual(hashed_paths.count(manifest_path.resolve()), 1)
            self.assertEqual(hashed_paths.count(draft_path.resolve()), 1)

    def test_strict_negative_requires_no_handedness(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {
                "crop_id": "negative:crop",
                "image": "frame.tiff",
                "palm_det_id": "negative:palm",
                "palm_valid": True,
                "palm_score": 0.8,
                "crop_path": "02_roi_crops/images/negative.png",
                "roi_rect": {"x_center": 127.5, "y_center": 127.5, "width": 255.0, "height": 255.0, "rotation_rad": 0.0},
                "roi_corners_px": [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]],
                "output_size": [256, 256],
            }
            draft = {
                **manifest,
                "hand_id": None,
                "hand_presence": {"present": False},
                "handedness": {"label": "unknown", "score": None},
                "landmarks_crop_norm": [],
                "landmarks_crop_px": [],
                "landmarks_image_px": [],
                "width": 256,
                "height": 256,
                "source_image_width": 1280,
                "source_image_height": 720,
            }
            cfg = {
                "image": {"width": 1280, "height": 720},
                "hand_roi": {"output_width": 256, "output_height": 256},
                "paths": {"roi_crops_dir": str(root / "02_roi_crops")},
                "cvat": {
                    "label_name": "hand_landmarks",
                    "no_hand_label_name": "no_hand",
                    "left_label_name": "Left",
                    "right_label_name": "Right",
                    "unknown_handedness_label_name": "unknown_handedness",
                    "ignore_for_training_label_name": "ignore_for_training",
                },
                "review": {
                    "strip_teacher_handedness": True,
                    "require_explicit_presence_decision": True,
                    "require_explicit_handedness_decision": True,
                },
            }
            xml_path = root / "negative.xml"
            export_cvat_xml([manifest], [draft], root, xml_path, cfg)
            rows, stats = import_cvat_xml(xml_path, [manifest], [draft], cfg)
            self.assertFalse(stats["errors"])
            self.assertEqual(rows[0]["finetune_review"]["presence_decision"], "no_hand")
            tree = ET.parse(xml_path)
            ET.SubElement(tree.getroot().find("image"), "tag", label="unknown_handedness", source="manual")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            _, conflicting = import_cvat_xml(xml_path, [manifest], [draft], cfg)
            self.assertTrue(conflicting["errors"])

    def test_selection_strict_cvat_publish_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "finetune" / "ft-test"
            parent = root / "parent"
            parent_image = parent / "02_roi_crops" / "images" / "crop.png"
            self.assertTrue(write_image(parent_image, np.full((256, 256), 127, dtype=np.uint8)))
            corners = [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]]
            manifest = {
                "crop_id": "parent:crop",
                "image": "frame.tiff",
                "palm_det_id": "parent:palm",
                "palm_valid": True,
                "palm_score": 0.9,
                "crop_path": str(parent_image),
                "roi_rect": {"x_center": 127.5, "y_center": 127.5, "width": 255.0, "height": 255.0, "rotation_rad": 0.0},
                "roi_corners_px": corners,
                "output_size": [256, 256],
            }
            norm = _landmarks()
            draft = {
                **manifest,
                "hand_id": "parent:crop:hand",
                "hand_presence": {"present": True},
                "handedness": {"label": "Left", "score": 0.99},
                "landmarks_crop_norm": norm,
                "landmarks_crop_px": [
                    {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
                    for point in norm
                ],
                "landmarks_image_px": [
                    {"id": point["id"], "x": point["x"] * 255.0, "y": point["y"] * 255.0}
                    for point in norm
                ],
                "width": 256,
                "height": 256,
                "source_image_width": 1280,
                "source_image_height": 720,
            }
            manifest_path = parent / "02_roi_crops" / "hand_roi_crops_manifest.jsonl"
            draft_path = parent / "02_roi_crops" / "hand_landmarks_autolabel_draft.jsonl"
            atomic_write_jsonl(manifest_path, [manifest])
            atomic_write_jsonl(draft_path, [draft])

            request_dir = workspace / "mining" / "hard_gold"
            request = request_dir / "selection_request.jsonl"
            atomic_write_jsonl(
                request,
                [
                    {
                        "source_kind": "reviewed_hard_gold",
                        "parent_dataset_id": "parent_ds",
                        "parent_source_crop_id": "parent:crop",
                        "parent_global_crop_id": "parent_ds:parent:crop",
                        "parent_manifest_path": str(manifest_path),
                        "parent_manifest_sha256": sha256_file(manifest_path),
                        "parent_draft_path": str(draft_path),
                        "parent_draft_sha256": sha256_file(draft_path),
                        "parent_crop_path": str(parent_image),
                        "image_sha256": sha256_file(parent_image),
                    }
                ],
            )
            request_row = json.loads(request.read_text(encoding="utf-8"))
            tampered_manifest_request = request_dir / "tampered_manifest.jsonl"
            atomic_write_jsonl(
                tampered_manifest_request,
                [{**request_row, "parent_manifest_sha256": "0" * 64}],
            )
            tampered_draft_request = request_dir / "tampered_draft.jsonl"
            atomic_write_jsonl(
                tampered_draft_request,
                [{**request_row, "parent_draft_sha256": "f" * 64}],
            )
            config_dir = root / "configs"
            config_dir.mkdir()
            registry_config = config_dir / "registry.yaml"
            registry_qc = root / "pretrain_qc"
            registry_config.write_text(
                "\n".join(
                    [
                        "sources:",
                        "  - dataset_id: parent_ds",
                        f'    root: "{parent.as_posix()}"',
                        "    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl",
                        "    pseudo_labels: 02_roi_crops/hand_landmarks_autolabel_draft.jsonl",
                        "    crop_images_dir: 02_roi_crops/images",
                        "outputs:",
                        "  pretrain:",
                        f'    qc_dir: "{registry_qc.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            registry_report = build_pretrain_source_registry(registry_config)
            self.assertEqual(registry_report["rows"], 1)
            registry_row = json.loads((registry_qc / "pretrain_source_registry.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(registry_row["global_crop_id"], "parent_ds:parent:crop")
            self.assertEqual(registry_row["image_sha256"], sha256_file(parent_image))

            config = config_dir / "finetune.yaml"
            config.write_text(
                "\n".join(
                    [
                        f'workspace_root: "{workspace.as_posix()}"',
                        "image: {width: 1280, height: 720, channels: 1, orientation: upright}",
                        "hand_roi: {output_width: 256, output_height: 256, scale_x: 1.8, scale_y: 1.8, shift_x: 0.0, shift_y: -0.1}",
                        "cvat:",
                        "  label_name: hand_landmarks",
                        "  no_hand_label_name: no_hand",
                        "  left_label_name: Left",
                        "  right_label_name: Right",
                        "  unknown_handedness_label_name: unknown_handedness",
                        "  ignore_for_training_label_name: ignore_for_training",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(GoldPipelineError):
                export_finetune_gold(
                    config,
                    source_id="tampered_manifest",
                    source_mode="selection_subset",
                    selection_request=tampered_manifest_request,
                )
            with self.assertRaises(GoldPipelineError):
                export_finetune_gold(
                    config,
                    source_id="tampered_draft",
                    source_mode="selection_subset",
                    selection_request=tampered_draft_request,
                )
            self.assertFalse((workspace / "cvat" / "tampered_manifest").exists())
            self.assertFalse((workspace / "cvat" / "tampered_draft").exists())
            export_finetune_gold(
                config,
                source_id="hard_gold",
                source_mode="selection_subset",
                selection_request=request,
            )
            task_root = workspace / "cvat" / "hard_gold"
            xml_path = task_root / "cvat_autolabel.xml"
            xml_text = xml_path.read_text(encoding="utf-8")
            self.assertNotIn('<tag label="Left"', xml_text)

            shutil.copy2(xml_path, task_root / "reviewed.xml")
            with self.assertRaises(GoldPipelineError):
                import_finetune_gold(config, source_id="hard_gold")

            tree = ET.parse(xml_path)
            image_element = tree.getroot().find("image")
            self.assertIsNotNone(image_element)
            ET.SubElement(image_element, "tag", label="unknown_handedness", source="manual")
            tree.write(task_root / "reviewed.xml", encoding="utf-8", xml_declaration=True)
            batch = import_all_finetune_gold(config)
            self.assertEqual([item["source_id"] for item in batch["published"]], ["hard_gold"])
            descriptor = json.loads(
                (workspace / "sources" / "gold" / "hard_gold" / "finetune_source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(descriptor["counts"]["included"], 1)
            self.assertEqual(descriptor["handedness_policy"], "optional_per_row")

            published_root = workspace / "sources" / "gold" / "hard_gold"
            gold_only_config = config_dir / "gold_only.yaml"
            gold_only_output = root / "gold_only_output"
            gold_only_config.write_text(
                "\n".join(
                    [
                        "schema_version: train_finalize_v1",
                        "sources:",
                        "  - dataset_id: hard_gold",
                        f'    root: "{published_root.as_posix()}"',
                        "    manifest: 02_roi_crops/hand_roi_crops_manifest.jsonl",
                        "    gold_labels: 03_reviewed/hand_landmarks_reviewed.jsonl",
                        "    crop_images_dir: 02_roi_crops/images",
                        "    source_mode: gold_only",
                        "    enabled_stages: [finetune]",
                        "    handedness_policy: optional_per_row",
                        "validation: {check_crop_images: true}",
                        "stages:",
                        "  finetune: {}",
                        "outputs:",
                        "  finetune:",
                        f'    labels_dir: "{(gold_only_output / "05_labels").as_posix()}"',
                        f'    qc_dir: "{(gold_only_output / "qc").as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            gold_only_report = finalize_training(gold_only_config, "finetune")
            self.assertEqual(gold_only_report["counts"]["included"], 1)

            native_task = export_finetune_gold(
                config,
                source_id="native_gold",
                source_mode="native_existing",
                raw_source_root=parent,
            )
            self.assertEqual(native_task["source_mode"], "native_existing")
            native_manifest = json.loads(
                (workspace / "cvat" / "native_gold" / "02_roi_crops" / "hand_roi_crops_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertIsNone(native_manifest["parent_global_crop_id"])
            self.assertEqual(native_manifest["native_source_crop_id"], "parent:crop")

            aggregate_config = config_dir / "aggregate.yaml"
            output = workspace / "hmlf_gold_merged"
            aggregate_config.write_text(
                "\n".join(
                    [
                        f'workspace_root: "{workspace.as_posix()}"',
                        "source_discovery:",
                        f'  root: "{(workspace / "sources" / "gold").as_posix()}"',
                        "  descriptor_name: finetune_source.json",
                        "outputs:",
                        f'  root: "{output.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            old_id = os.environ.get("HAND_FINETUNE_ID")
            os.environ["HAND_FINETUNE_ID"] = "ft-test"
            try:
                conflicting_root = workspace / "sources" / "gold" / "conflicting_gold"
                shutil.copytree(workspace / "sources" / "gold" / "hard_gold", conflicting_root)
                conflicting_labels = conflicting_root / "03_reviewed" / "hand_landmarks_reviewed.jsonl"
                conflict_row = json.loads(conflicting_labels.read_text(encoding="utf-8").strip())
                conflict_row["handedness"] = {"label": "Left", "score": None}
                conflict_row["finetune_review"]["handedness_decision"] = "Left"
                atomic_write_jsonl(conflicting_labels, [conflict_row])
                conflicting_descriptor_path = conflicting_root / "finetune_source.json"
                conflicting_descriptor = json.loads(conflicting_descriptor_path.read_text(encoding="utf-8"))
                conflicting_descriptor["source_id"] = "conflicting_gold"
                conflicting_descriptor["dataset_id"] = "conflicting_gold"
                conflicting_descriptor["artifacts"]["gold_labels"]["sha256"] = sha256_file(conflicting_labels)
                conflicting_descriptor_path.write_text(
                    json.dumps(conflicting_descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                with self.assertRaises(GoldPipelineError):
                    finalize_gold_aggregate(aggregate_config)
                shutil.rmtree(conflicting_root)
                aggregate = finalize_gold_aggregate(aggregate_config)
            finally:
                if old_id is None:
                    os.environ.pop("HAND_FINETUNE_ID", None)
                else:
                    os.environ["HAND_FINETUNE_ID"] = old_id
            self.assertEqual(aggregate["counts"], {"catalog": 1, "included": 1, "excluded": 0})
            canonical = json.loads(
                (output / "05_labels" / "hand_training_labels_finetune.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(canonical["finetune_review"]["presence_decision"], "hand")
            self.assertEqual(canonical["finetune_review"]["handedness_decision"], "unknown")
            self.assertEqual(canonical["handedness_loss_weight"], 0.0)
            self.assertTrue(canonical["finetune_review"]["source_descriptor_sha256"])


if __name__ == "__main__":
    unittest.main()
