from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hand_autolabel.quality_checks import RTMPOSE_CONNECTION_PAIRS
from tools.analyze_rtmpose_connection_lengths import (
    analyze_datasets,
    render_report,
    verify_config_thresholds,
)


class ConnectionLengthReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _points(scale: float) -> list[dict]:
        return [
            {
                "id": index,
                "x": 20.0 + index * scale,
                "y": 30.0 + index * scale * 0.5,
            }
            for index in range(21)
        ]

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _build_dataset(self) -> None:
        dataset_id = "demo-eval"
        variant = "v1"
        dataset_dir = self.root / "EValSource" / dataset_id
        sources = []
        for distance in ("near", "mid", "far"):
            source_id = f"white-{distance}-bright-random-val-s01-peak"
            labels_relpath = (
                f"EValSource/{dataset_id}/{source_id}/05_labels/{variant}/"
                "hand_evaluation_labels.jsonl"
            )
            gold_rows = [
                {
                    "crop_id": f"{distance}-one",
                    "human_reviewed": True,
                    "hand_presence": {"present": True},
                    "ignore_for_training": False,
                    "human_modified_landmark_ids": [],
                    "landmarks_crop_px": self._points(1.0),
                },
                {
                    "crop_id": f"{distance}-two",
                    "human_reviewed": True,
                    "hand_presence": {"present": True},
                    "ignore_for_training": False,
                    "human_modified_landmark_ids": [20],
                    "landmarks_crop_px": self._points(1.1),
                },
                {
                    "crop_id": f"{distance}-no-hand",
                    "human_reviewed": True,
                    "hand_presence": {"present": False},
                    "landmarks_crop_px": [],
                },
            ]
            self._write_jsonl(self.root / labels_relpath, gold_rows)
            draft_rows = [
                dict(row, source="rtmpose_m_hand5_onnx")
                for row in gold_rows[:2]
            ]
            draft_path = (
                dataset_dir
                / source_id
                / "02_roi_crops"
                / variant
                / "hand_landmarks_autolabel_draft.jsonl"
            )
            self._write_jsonl(draft_path, draft_rows)
            sources.append(
                {
                    "capture_source_id": source_id,
                    "published_variants": [
                        {
                            "proposal_variant": variant,
                            "labels_relpath": labels_relpath,
                        }
                    ],
                }
            )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "hlmf_dataset_v1",
                    "dataset_id": dataset_id,
                    "scope": "eval",
                    "capture_sources": sources,
                }
            ),
            encoding="utf-8",
        )

    def test_analysis_report_and_config_verification(self) -> None:
        self._build_dataset()
        analysis = analyze_datasets(self.root, [("demo-eval", "v1")])
        self.assertEqual(6, sum(row["valid_rows"] for row in analysis["sources"]))
        self.assertEqual(2, analysis["stats"]["near"][(19, 20)]["n"])
        self.assertGreater(analysis["stats"]["near"][(19, 20)]["variance"], 0.0)
        self.assertEqual(3, analysis["excluded"]["not_eligible_gold"])

        threshold_config = {
            distance: {
                f"{pair[0]}-{pair[1]}": analysis["thresholds"][distance][pair]
                for pair in RTMPOSE_CONNECTION_PAIRS
            }
            for distance in ("near", "mid", "far")
        }
        cfg = {
            "quality": {
                "rtmpose_train_connection_length_thresholds_px": threshold_config
            }
        }
        verify_config_thresholds(analysis, cfg)
        cfg["quality"]["rtmpose_train_connection_length_thresholds_px"]["near"][
            "19-20"
        ] += 1
        with self.assertRaisesRegex(ValueError, "config threshold mismatch"):
            verify_config_thresholds(analysis, cfg)

        report = render_report(analysis, "python -B tools/analyze.py ...")
        self.assertIn("## 结论", report)
        self.assertIn("<summary>完整 capture source 清单</summary>", report)
        self.assertIn("## near 统计", report)
        self.assertIn("| 连接 | N | 均值 | 方差 |", report)
        self.assertIn("## 运行与重新统计", report)
        self.assertNotIn("FullEnhanceVal0803", report)

    def test_partial_published_variant_preserves_unobserved_distance(self) -> None:
        self._build_dataset()
        manifest_path = self.root / "EValSource" / "demo-eval" / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for source in manifest["capture_sources"]:
            if "far" in source["capture_source_id"]:
                source["published_variants"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        fallback = {
            distance: {
                pair: 100 + index
                for index, pair in enumerate(RTMPOSE_CONNECTION_PAIRS)
            }
            for distance in ("near", "mid", "far")
        }
        analysis = analyze_datasets(
            self.root, [("demo-eval", "v1")], fallback_thresholds=fallback
        )

        self.assertEqual(["far"], analysis["preserved_distances"])
        self.assertEqual(0, analysis["stats"]["far"][(0, 1)]["n"])
        self.assertEqual(100, analysis["thresholds"]["far"][(0, 1)])
        report = render_report(analysis, "python -B tools/analyze.py ...")
        self.assertIn("阈值保留 YAML 中的历史值", report)
        self.assertIn("| 0-1 | 0 | — | — |", report)

    def test_missing_variant_is_rejected(self) -> None:
        self._build_dataset()
        fallback = {
            distance: {pair: 100 for pair in RTMPOSE_CONNECTION_PAIRS}
            for distance in ("near", "mid", "far")
        }
        with self.assertRaisesRegex(ValueError, "no published sources"):
            analyze_datasets(
                self.root,
                [("demo-eval", "missing")],
                fallback_thresholds=fallback,
            )


if __name__ == "__main__":
    unittest.main()
