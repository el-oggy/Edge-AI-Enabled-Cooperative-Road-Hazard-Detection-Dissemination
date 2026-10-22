"""Check saved native-run provenance, event order, and preserved project inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def audit(run_name="live_gui_final"):
    run = OUT / run_name
    summary, inputs = read(run / "summary.json"), read(run / "inputs.json")
    assert summary["status"] == "completed" and summary["sumo_exit_code"] == 0
    assert summary["authored_files_unchanged"] and summary["inference_executed"]
    assert summary["collision_vehicle_reports"] == summary["starting_teleports"] == 0
    records = [read(path) for path in (run / "inference").glob("*.json")]
    assert len(records) == 9
    model_counts, matrix, timings = Counter(), {}, []
    record_ids = {record["run_id"]: record for record in records}
    for record in records:
        model, source = record["model"], record["input"]
        model_id = model["source_model_id"]
        model_counts[model_id] += 1
        assert record["mode"] == "real_onnx" and record["inference_executed"] and not record["mock_fallback"]
        assert sha(Path(model["path"])) == model["sha256"] == inputs["models"][model_id]["sha256"]
        assert sha(Path(source["path"])) == source["sha256"]
        raw = np.load(run / "inference" / (record["run_id"] + "_raw.npy"), allow_pickle=False)
        assert list(raw.shape) == [1, 5, 12096] and raw.dtype == np.float32
        assert hashlib.sha256(raw.tobytes()).hexdigest() == record["raw_output"]["sha256"]
        assert source["tensor_shape"] == [1, 3, 768, 768]
        assert record["runtime"]["providers"] == ["CPUExecutionProvider"]
        assert record["thresholds"]["confidence"] == .35
        assert (run / "inference" / (record["run_id"] + "_annotated.jpg")).is_file()
        matrix.setdefault(Path(source["path"]).name, {})[model_id] = [d["confidence"] for d in record["detections"]]
        timings.append(record["timings_ms"]["inference"])
    assert model_counts == Counter(pothole=3, construction=3, accident_obstacle=3)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    reports, broadcasts, warnings, requests, responses = {}, {}, {}, {}, {}
    verified_bypasses, chain_examples = 0, []
    for event in events:
        kind, now = event["kind"], event["sim_time_s"]
        if kind == "vehicle_report":
            record = record_ids[event["prediction_run_id"]]
            assert record["detections"] and event["source_model_id"] == record["model"]["source_model_id"]
            elapsed = record["completed_wall_s"] - record["encounter"]["capture_wall_s"]
            assert event["available_at_s"] + 1e-6 >= event["capture_sim_s"] + elapsed + .2
            assert event["available_at_s"] + 1e-6 >= now + .2
            assert abs(event["expires_at_s"] - event["available_at_s"] - 300) < 1e-6
            reports[event["id"]] = event
        elif kind == "rsu_broadcast":
            assert event["id"] in reports and now + 1e-6 >= event["available_at_s"]
            broadcasts[event["id"]] = event
        elif kind == "vehicle_warning":
            report = broadcasts[event["warning_id"]]
            assert report["available_at_s"] <= now < report["expires_at_s"]
            warnings[event["vehicle_id"], event["warning_id"]] = event
        elif kind == "avoidance_request":
            key = event["vehicle_id"], event["warning_id"]
            assert key in warnings and now >= warnings[key]["sim_time_s"]
            assert 0 < event["distance_m"] <= 80
            requests[key] = event
        elif kind == "lane_response_observed":
            key = event["vehicle_id"], event["warning_id"]
            assert key in requests and now > requests[key]["sim_time_s"]
            assert event["lane_index"] == requests[key]["requested_lane"]
            responses[key] = event
        elif kind == "observed_pass" and event["warned_request_and_response_preceded"]:
            key = event["vehicle_id"], event["warning_id"]
            assert key in responses and responses[key]["sim_time_s"] <= now
            assert now - requests[key]["sim_time_s"] <= 30 and event["bypass"]
            assert now < reports[event["warning_id"]]["expires_at_s"]
            verified_bypasses += 1
            chain_examples.append({"vehicle": key[0], "warning_id": key[1],
                "warning_s": warnings[key]["sim_time_s"], "request_s": requests[key]["sim_time_s"],
                "response_s": responses[key]["sim_time_s"], "bypass_s": now})
    assert verified_bypasses == summary["counts"]["avoided"] > 0
    screenshots = []
    for filename in summary["screenshots"]:
        with Image.open(filename) as image:
            assert image.convert("RGB").getbbox()
            screenshots.append({"path": filename, "size": image.size, "sha256": sha(Path(filename))})
    scene = read(ROOT / "simulation/cooperative_loop_1km/scene.json")
    overview = next(item for item in summary["screenshot_contexts"] if item["name"] == "overview")
    (x0, y0), (x1, y1) = overview["boundary"]
    tx0, ty0, tx1, ty1 = scene["bounds"]
    # Native GUI evidence must contain the whole authored scene and use the view.
    if run_name == "live_gui_presented":
        assert x0 <= tx0 and y0 <= ty0 and x1 >= tx1 and y1 >= ty1
        assert max((tx1-tx0)/(x1-x0), (ty1-ty0)/(y1-y0)) >= .9
    before = read(OUT / "preservation_before.json")
    changed = [path for path, value in before["files"].items() if sha(ROOT / path) != value]
    assert set(changed) == {"Run SUMO Simulation.cmd", "Run SUMO Simulation.vbs"}
    assert git("branch", "--show-current") == before["branch"] and git("rev-parse", "HEAD") == before["head"]
    assert not git("diff", "--cached", "--name-only")
    assert git("diff", "--name-only") == ".gitignore"
    added_md = git("ls-files", "--others", "--exclude-standard", "*.md")
    # Historical skill/research Markdown predates this implementation.
    assert not any(name.endswith(".md") for name in inputs["input_sha256"])
    result = {"passed": True, "run": run_name, "model_forward_counts": dict(model_counts),
        "predictions": matrix, "actual_inference_ms_range": [min(timings), max(timings)],
        "verified_report_broadcast_warning_request_response_bypass_sequences": verified_bypasses,
        "sequence_examples": chain_examples[:4], "reports": len(reports), "broadcasts": len(broadcasts),
        "summary": summary, "screenshots": screenshots, "baseline_files_checked": len(before["files"]),
        "intentional_baseline_changes": changed, "branch": before["branch"], "head": before["head"],
        "historical_untracked_markdown": added_md.splitlines(),
        "limitations": ["Event ordering verifies a sequence, not counterfactual collision reduction.",
            "Static manually mapped photographs, assumed radio coverage, desktop CPU timings."]}
    (OUT / ("delivery_audit_" + run_name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("summary", "historical_untracked_markdown", "screenshots")}, indent=2))


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv) > 1 else "live_gui_final")
