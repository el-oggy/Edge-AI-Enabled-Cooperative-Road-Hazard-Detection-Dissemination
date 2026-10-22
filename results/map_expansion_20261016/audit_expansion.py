"""Audit saved expanded-map execution and preservation evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

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


def main():
    scene = read(ROOT / "simulation/cooperative_loop_1km/scene.json")
    run = OUT / "live_gui_final"
    summary, inputs = read(run / "summary.json"), read(run / "inputs.json")
    expected_ids = {h["id"] for h in scene["hazards"]}
    rsu_ids = {r["id"] for r in scene["rsus"]}
    assert summary["status"] == "completed" and summary["sumo_exit_code"] == 0
    assert summary["authored_files_unchanged"] and summary["inference_executed"]
    assert summary["collision_vehicle_reports"] == summary["starting_teleports"] == 0
    assert set(summary["processed_hazards"]) == expected_ids and len(expected_ids) == 9
    assert len(rsu_ids) == 8 and summary["bridge_observed_elevations_m"] == {"upper":8., "lower":0.}
    assert all(summary["bridge_observed_vehicle_counts"].values())
    records = [read(path) for path in (run / "inference").glob("*.json")]
    record_ids, cases, times = {}, Counter(), []
    predictions = {}
    for record in records:
        record_ids[record["run_id"]] = record
        model, context = record["model"], record["encounter"]
        mid = model["source_model_id"]
        cases[context["hazard_id"],mid] += 1
        assert record["mode"] == "real_onnx" and record["inference_executed"] and not record["mock_fallback"]
        assert sha(Path(model["path"])) == model["sha256"] == inputs["models"][mid]["sha256"]
        assert sha(Path(record["input"]["path"])) == record["input"]["sha256"]
        assert record["input"]["tensor_shape"] == [1,3,768,768]
        raw = np.load(run / "inference" / (record["run_id"] + "_raw.npy"), allow_pickle=False)
        assert raw.shape == (1,5,12096) and raw.dtype == np.float32 and np.isfinite(raw).all()
        assert hashlib.sha256(raw.tobytes()).hexdigest() == record["raw_output"]["sha256"]
        assert record["thresholds"]["confidence"] == .35
        assert record["runtime"]["providers"] == ["CPUExecutionProvider"]
        predictions.setdefault(context["hazard_id"], {})[mid] = [d["confidence"] for d in record["detections"]]
        times.append(record["timings_ms"]["inference"])
    assert all(cases[hid,mid] >= 1 for hid in expected_ids for mid in ("pothole","construction","accident_obstacle"))
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    reports, broadcasts, warnings, requests, responses = {}, {}, {}, {}, {}
    bypasses, chains = 0, []
    for event in events:
        kind, now = event["kind"], event["sim_time_s"]
        if kind == "vehicle_report":
            record = record_ids[event["prediction_run_id"]]
            assert record["detections"] and event["source_model_id"] == record["model"]["source_model_id"]
            assert set(event["rsu_ids"]) == rsu_ids
            assert now - record["encounter"]["capture_sim_s"] <= 30
            elapsed = record["completed_wall_s"] - record["encounter"]["capture_wall_s"]
            assert event["available_at_s"] + 1e-6 >= event["capture_sim_s"] + elapsed + .2
            assert event["available_at_s"] + 1e-6 >= now + .2
            reports[event["id"]] = event
        elif kind == "rsu_broadcast":
            assert event["id"] in reports and now + 1e-6 >= event["available_at_s"]
            broadcasts[event["id"]] = event
        elif kind == "vehicle_warning":
            report = broadcasts[event["warning_id"]]
            assert report["available_at_s"] <= now < report["expires_at_s"]
            warnings[event["vehicle_id"],event["warning_id"]] = event
        elif kind == "avoidance_request":
            key = event["vehicle_id"],event["warning_id"]
            assert key in warnings and warnings[key]["sim_time_s"] <= now
            assert 0 < event["distance_m"] <= 80
            requests[key] = event
        elif kind == "lane_response_observed":
            key = event["vehicle_id"],event["warning_id"]
            assert key in requests and requests[key]["sim_time_s"] < now
            assert requests[key]["requested_lane"] == event["lane_index"]
            responses[key] = event
        elif kind == "observed_pass" and event["warned_request_and_response_preceded"]:
            key = event["vehicle_id"],event["warning_id"]
            assert key in responses and responses[key]["sim_time_s"] <= now
            assert now - requests[key]["sim_time_s"] <= 30 and event["bypass"]
            assert now < reports[event["warning_id"]]["expires_at_s"]
            bypasses += 1
            chains.append({"vehicle":key[0],"hazard":event["hazard_id"],"warning_s":warnings[key]["sim_time_s"],
                "request_s":requests[key]["sim_time_s"],"response_s":responses[key]["sim_time_s"],"pass_s":now})
    assert {r["hazard_id"] for r in reports.values()} == expected_ids
    assert bypasses == summary["counts"]["avoided"] > 0
    screens = []
    for filename in summary["screenshots"]:
        with Image.open(filename) as frame:
            assert frame.convert("RGB").getbbox()
            screens.append({"path":filename,"dimensions":frame.size,"sha256":sha(Path(filename))})
    overview = next(c for c in summary["screenshot_contexts"] if c["name"] == "overview")
    (x0,y0),(x1,y1) = overview["boundary"]
    tx0,ty0,tx1,ty1 = scene["bounds"]
    assert x0 <= tx0 and y0 <= ty0 and x1 >= tx1 and y1 >= ty1
    traffic = read(OUT / "traffic_1200/summary.json")
    assert traffic["status"] == "completed" and traffic["sumo_exit_code"] == 0
    assert traffic["collision_vehicle_reports"] == traffic["starting_teleports"] == 0
    samples = [json.loads(line) for line in (OUT / "traffic_1200/traffic.jsonl").read_text().splitlines()]
    assert all(row["active_vehicles"] == 144 for row in samples if row["sim_time_s"] >= 220)
    before = read(OUT / "preservation_before.json")
    changed = [p for p,value in before["files"].items() if sha(ROOT / p) != value]
    assert all(p.startswith("simulation/cooperative_loop_1km/") or p == "tests/test_cooperative_loop.py" for p in changed)
    assert git("branch","--show-current") == before["branch"] and git("rev-parse","HEAD") == before["head"]
    assert not git("diff","--cached","--name-only") and git("diff","--name-only") == ".gitignore"
    with zipfile.ZipFile(OUT / "previous_1km_demo.zip") as backup:
        assert backup.testzip() is None
        for filename in backup.namelist():
            assert hashlib.sha256(backup.read(filename)).hexdigest() == before["files"][filename]
    result = {"passed":True,"scene":scene["scene"],"actual_loop_length_m":scene["main_route_length_m"],
        "rsus":len(rsu_ids),"hazard_locations":len(expected_ids),"forward_records":len(records),
        "forward_counts_by_model":dict(Counter(r["model"]["source_model_id"] for r in records)),
        "inference_ms_range":[min(times),max(times)],"reports":len(reports),"broadcasts":len(broadcasts),
        "warnings":len([e for e in events if e["kind"]=="vehicle_warning"]),"verified_warned_bypasses":bypasses,
        "sequence_examples":chains[:5],"predictions":predictions,"summary":summary,"traffic_summary":traffic,
        "screenshots":screens,"baseline_files_checked":len(before["files"]),"intentional_changes":changed,
        "original_1km_backup_verified":True,"head":before["head"],"branch":before["branch"],
        "limits":["Three manually mapped photographs reused across nine simulated locations.",
            "Full-map modeled radio coverage; no physical LoRa range or latency qualification.",
            "Warned bypasses verify event order, not counterfactual accident reduction."]}
    (OUT / "delivery_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("summary","traffic_summary","screenshots","predictions")},indent=2))


if __name__ == "__main__":
    main()
