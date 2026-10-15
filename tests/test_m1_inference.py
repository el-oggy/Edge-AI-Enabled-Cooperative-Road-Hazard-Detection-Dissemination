"""Synthetic unit fixtures verify implementation, not trained-model accuracy."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import edge_rpi_hazard_inference as edge
from pfpd_perception.model_registry import MODELS, validate_session_contract


def info(model="pothole"):
    return {**MODELS[model], "source_model_id": model}


def predictions(rows):
    return np.asarray(rows, dtype=np.float64).T[None]


class ImageAndDecodeTests(unittest.TestCase):
    def test_rgb_normalization_contiguous_and_padding(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[:] = [0, 128, 255]
        tensor, shape, transform, original = edge.preprocess_image(image, input_size=8)
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 3, 3], [1, 128 / 255, 0])
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
        self.assertEqual(shape, (2, 4))
        self.assertEqual(transform, (2.0, 0, 2))
        self.assertIs(original, image)

    def test_odd_integer_borders_inverse_non_square(self):
        image = np.zeros((3, 7, 3), dtype=np.uint8)
        tensor, shape, transform, _ = edge.preprocess_image(image, (8, 8))
        scale, left, top = transform
        self.assertEqual(top, 2)  # bottom is 3, not an assumed 2.5/2.5.
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        box = [1.25, 0.6, 5.75, 2.4]
        x1, y1, x2, y2 = box
        row = [((x1+x2)/2)*scale+left, ((y1+y2)/2)*scale+top,
               (x2-x1)*scale, (y2-y1)*scale, 0.75]
        detections = edge.postprocess(predictions([row]), shape, transform, info())
        np.testing.assert_allclose(detections[0]["bbox_xyxy_float"], box, atol=1e-12)
        self.assertEqual(detections[0]["box"], [1, 1, 6, 2])

    def test_missing_corrupt_empty_and_wrong_dtype_images(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                edge.preprocess_image(missing)
            corrupt = Path(directory) / "invalid.png"
            corrupt.write_text("not a PNG", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decode"):
                edge.preprocess_image(corrupt)
        for image in [np.zeros((0, 5, 3), np.uint8), np.zeros((5, 5), np.uint8),
                      np.zeros((5, 5, 3), np.float32), None]:
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                edge.preprocess_image(image)

    def test_source_semantics_raw_class_zero_for_each_model(self):
        raw = predictions([[10, 10, 8, 8, 0.7]])
        for model in MODELS:
            det = edge.postprocess(raw, (20, 20), (1, 0, 0), info(model))[0]
            self.assertEqual(det["class_id"], 0)
            self.assertEqual(det["raw_class_id"], 0)
            self.assertEqual(det["class_name"], MODELS[model]["name"])
            self.assertEqual(det["source_model_id"], model)

    def test_nms_keeps_strongest_and_disjoint_boxes(self):
        raw = predictions([[10, 10, 10, 10, 0.6], [10.1, 10.1, 10, 10, 0.9],
                           [40, 40, 5, 5, 0.8]])
        detections = edge.postprocess(raw, (50, 50), (1, 0, 0), info())
        self.assertEqual([det["proposal_index"] for det in detections], [1, 2])
        self.assertEqual([det["confidence"] for det in detections], [0.9, 0.8])

    def test_float_nms_preserves_distinct_subpixel_boxes(self):
        raw = predictions([[2.1, 2, 1.1, 2, .9], [2.9, 2, 1.1, 2, .8]])
        detections = edge.postprocess(raw, (10, 10), (1, 0, 0), info(), iou_thresh=.45)
        self.assertEqual(len(detections), 2)
        self.assertAlmostEqual(detections[0]["bbox_xyxy_float"][0], 1.55)

    def test_inclusive_threshold_and_no_objectness_multiplication(self):
        detections = edge.postprocess(predictions([[10, 10, 4, 4, .35]]),
                                      (20, 20), (1, 0, 0), info(), conf_thresh=.35)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["confidence"], .35)
        self.assertEqual(edge.postprocess(predictions([[10, 10, 4, 4, .3499]]),
                                          (20, 20), (1, 0, 0), info()), [])

    def test_clip_to_image_and_reject_degenerate_boxes(self):
        raw = predictions([[0, 0, 8, 8, .9], [50, 50, 4, 4, .8],
                           [10, 10, 0, 5, .7], [10, 10, -2, 5, .6]])
        detections = edge.postprocess(raw, (20, 20), (1, 0, 0), info())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box"], [0, 0, 4, 4])

    def test_singleton_transposed_and_empty_output(self):
        row = [[10, 10, 4, 4, .7]]
        for raw in (predictions(row), np.asarray(row), np.asarray(row).T):
            with self.subTest(shape=raw.shape):
                self.assertEqual(len(edge.postprocess(raw, (20, 20), (1, 0, 0), info())), 1)
        self.assertEqual(edge.postprocess(np.empty((1, 5, 0)), (20, 20), (1, 0, 0), info()), [])

    def test_bad_outputs_and_missing_semantics_fail(self):
        for raw in [np.zeros((2, 5, 1)), np.zeros((1, 6, 12)),
                    predictions([[10, 10, 5, 5, float("nan")]]),
                    predictions([[float("inf"), 10, 5, 5, .9]]),
                    predictions([[10, 10, 5, 5, 1.1]])]:
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                edge.postprocess(raw, (20, 20), (1, 0, 0), info())
        with self.assertRaisesRegex(ValueError, "semantics"):
            edge.postprocess(predictions([[10, 10, 5, 5, .9]]), (20, 20), (1, 0, 0))

    def test_invalid_thresholds_and_scale_fail(self):
        for kwargs in ({"conf_thresh": float("nan")}, {"iou_thresh": 1.1}):
            with self.assertRaises(ValueError):
                edge.postprocess(predictions([[10, 10, 5, 5, .9]]), (20, 20), (1, 0, 0), info(), **kwargs)
        with self.assertRaises(ValueError):
            edge.postprocess(predictions([[10, 10, 5, 5, .9]]), (20, 20), (0, 0, 0), info())


class IdentityAndFailureTests(unittest.TestCase):
    def test_actual_reviewed_artifact_hashes_and_relative_path(self):
        for model, expected in MODELS.items():
            record = edge.resolve_hazard_info(expected["path"])
            self.assertEqual(record["sha256"], expected["sha256"])
            self.assertEqual(record["source_model_id"], model)
            self.assertTrue(Path(record["model_path"]).is_absolute())

    def test_unreviewed_or_missing_model_never_guessed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            missing = Path(directory) / "missing_pothole.onnx"
            with self.assertRaises(FileNotFoundError):
                edge.resolve_hazard_info(missing)
            missing.write_bytes(b"untrusted artifact named pothole")
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                edge.resolve_hazard_info(missing)

    def fake_session(self, names="{0: 'pothole'}", shape=(1, 3, 768, 768)):
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=shape)],
            get_outputs=lambda: [SimpleNamespace(name="output0", type="tensor(float)", shape=(1, 5, 12096))],
            get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"names": names, "task": "detect", "end2end": "False"}))

    def test_metadata_shape_and_unsafe_literal_rejected(self):
        for session in [self.fake_session(names="{0: 'construction_hazard'}"),
                        self.fake_session(names="__import__('os').system('never')"),
                        self.fake_session(shape=(1, 3, 640, 640)),
                        self.fake_session(shape=(1, 3, "height", "width"))]:
            with self.assertRaises(ValueError):
                validate_session_contract(session, info())
        self.assertEqual(validate_session_contract(self.fake_session(), info())["names"], "{0: 'pothole'}")

    def run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = edge.main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_missing_runtime_nonzero_no_mock_fallback(self):
        with patch.object(edge, "ort", None):
            code, stdout, stderr = self.run_main(["--image", "data/test_user_pothole.png"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        error = json.loads(stderr)
        self.assertEqual(error["error_type"], "RuntimeError")
        self.assertIsNone(error["inference_result"])
        self.assertFalse(error["mock_fallback"])

    def test_inference_exception_propagates_nonzero(self):
        fake = self.fake_session()
        fake.run = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("actual runtime failed"))
        with patch.object(edge, "create_session", return_value=(fake, {})):
            code, stdout, stderr = self.run_main(["--image", "data/test_user_pothole.png"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("actual runtime failed", stderr)
        self.assertFalse(json.loads(stderr)["mock_fallback"])

    def test_missing_image_nonzero(self):
        code, stdout, stderr = self.run_main(["--image", "data/M1_ABSENT_INPUT.png"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error_type"], "FileNotFoundError")

    def test_missing_opencv_fails_clearly(self):
        with patch.object(edge, "cv2", None), self.assertRaisesRegex(RuntimeError, "OpenCV"):
            edge.preprocess_image(np.zeros((2, 2, 3), np.uint8))

    def test_mock_is_explicit_and_real_combination_rejected(self):
        code, stdout, stderr = self.run_main(["--simulate-stream"])
        self.assertEqual(code, 0)
        result = json.loads(stdout)
        self.assertEqual(result["mode"], "mock")
        self.assertFalse(result["inference_executed"])
        self.assertEqual([len(frame["confirmed"]) for frame in result["frames"]], [0, 0, 1, 0, 0])
        with self.assertRaises(SystemExit) as raised:
            self.run_main(["--simulate-stream", "--image", "data/test_user_pothole.png"])
        self.assertEqual(raised.exception.code, 2)

    def test_benchmark_output_combination_fails_before_session(self):
        with patch.object(edge, "create_session") as create:
            with self.assertRaises(SystemExit) as raised:
                self.run_main(["--benchmark", "--output-dir", "results/forbidden"])
        self.assertEqual(raised.exception.code, 2)
        create.assert_not_called()

    def test_existing_json_is_preserved_before_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "existing.json"
            path.write_bytes(b"original evidence")
            with patch.object(edge, "create_session") as create:
                code, stdout, stderr = self.run_main(["--image", "data/test_user_pothole.png", "--json-out", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(stderr)["error_type"], "FileExistsError")
            self.assertEqual(path.read_bytes(), b"original evidence")
            create.assert_not_called()

    def test_single_image_beacon_does_not_invent_confirmation_or_subtype(self):
        det = edge.postprocess(predictions([[10, 10, 5, 5, .9]]), (20, 20), (1, 0, 0), info("accident_obstacle"))[0]
        beacon = edge.generate_etsi_denm_beacon("TEST", [det])
        self.assertFalse(beacon["standards_compliant"])
        self.assertFalse(beacon["transmitted"])
        hazard = beacon["denm"]["situation"]["hazards"][0]
        self.assertIsNone(hazard["persistenceFrames"])
        self.assertIsNone(hazard["trackID"])
        self.assertIsNone(hazard["causeCode"])
        self.assertEqual(hazard["validationStatus"], "SINGLE_FRAME_UNCONFIRMED")
        self.assertEqual(hazard["severity"], "UNASSESSED")


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.tracker = edge.TemporalHazardTracker()
        self.det = {"class_id": 0, "class_name": "pothole", "confidence": .8,
                    "source_model_id": "pothole", "box": [10, 10, 30, 30]}

    def update(self, frame, detections=None, capture=None, source="camera", model="pothole"):
        return self.tracker.update([self.det] if detections is None else detections,
                                   frame_id=frame, capture_time_s=frame*.1 if capture is None else capture,
                                   source_id=source, source_model_id=model)

    def test_consecutive_frames_confirm_with_conservative_score(self):
        self.assertEqual(self.update(0), [])
        self.assertEqual(self.update(1, [{**self.det, "confidence": .7}]), [])
        confirmed = self.update(2, [{**self.det, "confidence": .9}])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["persistence_hits"], 3)
        self.assertEqual(confirmed[0]["confidence"], .7)
        self.assertEqual([hit["frame_id"] for hit in confirmed[0]["evidence"]], [0, 1, 2])

    def test_miss_resets_and_no_stale_confirmation(self):
        self.update(0); self.update(1)
        self.assertEqual(self.update(2, []), [])
        self.assertEqual(self.update(3), [])
        self.assertEqual(self.update(4), [])
        self.assertEqual(len(self.update(5)), 1)
        self.assertEqual(self.update(6, []), [])

    def test_repeat_frame_id_out_of_order_or_repeat_timestamp_rejected(self):
        self.update(1)
        for frame, capture in [(1, .2), (0, .2), (2, .1), (2, 0), (2, float("nan"))]:
            with self.subTest(frame=frame, time=capture), self.assertRaises(ValueError):
                self.update(frame, capture=capture)
        self.assertEqual(self.update(2), [])

    def test_gap_resets_streak_without_forcing_contiguous_input_ids(self):
        self.update(0); self.update(10, capture=.2)
        self.assertEqual(self.update(20, capture=1.5), [])
        self.assertEqual(self.update(21, capture=1.6), [])
        self.assertEqual(len(self.update(22, capture=1.7)), 1)

    def test_elapsed_expiry_uses_new_track_identity(self):
        self.update(0); self.update(1)
        old_id = self.tracker.tracks[0]["id"]
        self.assertEqual(self.update(2, capture=3), [])
        self.assertNotEqual(self.tracker.tracks[0]["id"], old_id)

    def test_sources_and_models_cannot_share_hits(self):
        self.update(0); self.update(1)
        other = {**self.det, "source_model_id": "construction", "class_name": "construction_hazard"}
        self.assertEqual(self.update(0, [other], source="camera", model="construction"), [])
        self.assertEqual(self.update(0, source="camera2"), [])
        self.assertEqual(len(self.update(2)), 1)

    def test_one_to_one_matching_cannot_count_two_same_frame_hits(self):
        self.update(0)
        self.assertEqual(self.update(1, [self.det, self.det]), [])
        self.assertEqual([track["hits"] for track in self.tracker.tracks], [2, 1])
        confirmed = self.update(2, [self.det])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["persistence_hits"], 3)

    def test_invalid_detection_does_not_mutate_state(self):
        self.update(0)
        baseline = copy.deepcopy(self.tracker.__dict__)
        bad_detections = [{"box": [1, 1, 2, 2], "confidence": .8},
                          {**self.det, "box": [1, 1, 1, 2]},
                          {**self.det, "source_model_id": "construction"},
                          {**self.det, "class_name": "road_accident_obstacle"},
                          {**self.det, "raw_class_id": 1},
                          {**self.det, "confidence": float("nan")},
                          {**self.det, "class_id": 1}]
        for bad in bad_detections:
            with self.assertRaises((ValueError, KeyError)):
                self.update(1, [bad])
            self.assertEqual(self.tracker.__dict__, baseline)
        self.assertEqual(self.update(1), [])

    def test_unknown_model_cannot_mutate_tracker(self):
        baseline = copy.deepcopy(self.tracker.__dict__)
        with self.assertRaises(ValueError):
            self.tracker.update([], frame_id=0, capture_time_s=0.0,
                                source_id="camera", source_model_id="unknown")
        self.assertEqual(self.tracker.__dict__, baseline)

    def test_tracker_invalid_configuration_and_identity(self):
        for kwargs in [{"min_hits": 1.5}, {"min_hits": True}, {"max_misses": 0},
                       {"max_misses": True}, {"max_capture_gap_s": float("inf")}]:
            with self.assertRaises(ValueError):
                edge.TemporalHazardTracker(**kwargs)
        for source in [[], 1, "", " "]:
            with self.assertRaises(ValueError):
                self.update(0, source=source)

    def test_frame_expiry_and_semantic_mismatch(self):
        self.tracker = edge.TemporalHazardTracker(max_misses=2)
        self.update(0); self.update(1, [])
        self.assertEqual(self.update(2, []), [])
        self.assertEqual(self.tracker.tracks, [])


if __name__ == "__main__":
    unittest.main()
