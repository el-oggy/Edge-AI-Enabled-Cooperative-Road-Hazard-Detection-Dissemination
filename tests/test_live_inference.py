"""Contract tests for the three-model real ONNX adapter; sessions are mocked."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pfpd_perception import live_inference
from pfpd_perception.model_registry import MODELS


class FakeSession:
    def __init__(self, model_id, empty=False, fail=False):
        self.model_id = model_id
        self.empty = empty
        self.fail = fail
        self.calls = 0

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic runtime failure")
        if self.empty:
            return [np.zeros((1, 5, 12096), dtype=np.float32)]
        raw = np.zeros((1, 5, 12096), dtype=np.float32)
        raw[0, :, 0] = [384, 384, 120, 100, 0.8]
        return [raw]


class ThreeModelInferenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.sessions = {model_id: FakeSession(model_id) for model_id in MODELS}
        self.image_path = ROOT / "tests" / "synthetic_fixture.png"
        edge = live_inference.edge
        patches = [
            patch.object(edge, "cv2", SimpleNamespace(__version__="test-cv2")),
            patch.object(edge, "ort", SimpleNamespace(__version__="test-ort")),
            patch.object(edge, "project_path", side_effect=lambda path: Path(path)),
            patch.object(edge, "sha256_file", return_value="synthetic-input-sha256"),
            patch.object(live_inference, "sha256_file", return_value="synthetic-input-sha256"),
            patch.object(edge, "preprocess_image",
                         return_value=(np.zeros((1, 3, 768, 768), np.float32),
                                       (48, 64), (12.0, 0, 96), self.image)),
            patch.object(edge, "_annotate", side_effect=lambda image, detections: image.copy()),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_adapter(self, adapter, encounter=None):
        return list(adapter.infer_image(self.image_path, encounter=encounter))

    @staticmethod
    def resolve_identity(model_path):
        model_id = next(key for key in MODELS if Path(model_path).name.startswith(key))
        return {**MODELS[model_id], "source_model_id": model_id,
                "model_path": str(model_path)}

    def test_constructs_three_authenticated_sessions_once_and_reuses_them(self):
        session_calls = []

        def create_session(model_path, hazard_info=None, intra_op_threads=4):
            model_id = hazard_info["source_model_id"]
            session_calls.append((model_id, intra_op_threads))
            return self.sessions[model_id], {"input": "images", "output": "output0"}

        with patch.object(live_inference.edge, "create_session", side_effect=create_session), \
                patch.object(live_inference, "resolve_hazard_info", side_effect=self.resolve_identity):
            adapter = live_inference.ThreeModelInference(conf=0.35, iou=0.45, threads=4)
            first = self.run_adapter(adapter, {"id": "enc-1"})
            second = self.run_adapter(adapter, {"id": "enc-2"})

        self.assertEqual(len(session_calls), 3)
        self.assertEqual({model for model, _ in session_calls}, set(MODELS))
        self.assertTrue(all(threads == 4 for _, threads in session_calls))
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertTrue(all(session.calls == 2 for session in self.sessions.values()))
        for records, encounter_id in ((first, "enc-1"), (second, "enc-2")):
            self.assertEqual({r["model"]["source_model_id"] for r in records}, set(MODELS))
            for record in records:
                self.assertEqual(record["mode"], "real_onnx")
                self.assertTrue(record["inference_executed"])
                self.assertFalse(record["mock_fallback"])
                self.assertEqual(record["encounter"]["id"], encounter_id)
                self.assertEqual(record["_raw_output_array"].shape, (1, 5, 12096))
                self.assertTrue(np.isfinite(record["_raw_output_array"]).all())
                self.assertIn("timings_ms", record)
                self.assertTrue(any(key in record for key in ("input_metadata", "input")))
                self.assertIn("detections", record)
                self.assertIn("_annotation", record)
                self.assertIn("completed_wall_s", record)

    def test_empty_raw_output_remains_no_detections(self):
        self.sessions["construction"].empty = True

        def create_session(model_path, hazard_info=None, intra_op_threads=4):
            model_id = hazard_info["source_model_id"]
            return self.sessions[model_id], {}

        with patch.object(live_inference.edge, "create_session", side_effect=create_session), \
                patch.object(live_inference, "resolve_hazard_info", side_effect=self.resolve_identity):
            records = self.run_adapter(live_inference.ThreeModelInference())
        construction = next(r for r in records if r["model"]["source_model_id"] == "construction")
        self.assertEqual(construction["detections"], [])
        self.assertEqual(construction["_raw_output_array"].shape, (1, 5, 12096))

    def test_session_run_failure_propagates_without_mock_record(self):
        self.sessions["accident_obstacle"].fail = True

        def create_session(model_path, hazard_info=None, intra_op_threads=4):
            model_id = hazard_info["source_model_id"]
            return self.sessions[model_id], {}

        with patch.object(live_inference.edge, "create_session", side_effect=create_session), \
                patch.object(live_inference, "resolve_hazard_info", side_effect=self.resolve_identity):
            adapter = live_inference.ThreeModelInference()
            with self.assertRaisesRegex(RuntimeError, "synthetic runtime failure"):
                self.run_adapter(adapter)

    def test_unregistered_model_identity_is_not_guessed(self):
        with patch.object(live_inference, "resolve_hazard_info",
                          side_effect=ValueError("unreviewed model")):
            with self.assertRaisesRegex(ValueError, "unreviewed"):
                live_inference.ThreeModelInference()


if __name__ == "__main__":
    unittest.main()
