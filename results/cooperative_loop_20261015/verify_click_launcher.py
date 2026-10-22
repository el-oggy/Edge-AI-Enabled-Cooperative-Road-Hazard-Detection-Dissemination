"""Bounded desktop verification of this task's VBS launcher; closes its own window."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / ("click_launcher_check_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
RUNS = ROOT / "results" / "cooperative_runs"
before = set(RUNS.glob("*"))
started = time.monotonic()
launch = subprocess.run(["wscript.exe", str(ROOT / "Run SUMO Simulation.vbs")],
                        cwd=ROOT.parent.parent, timeout=10, capture_output=True)
run_dir = None
for _ in range(300):
    new = [p for p in RUNS.glob("*") if p not in before and (p / "inputs.json").exists()]
    if len(new) == 1:
        run_dir = new[0]
        break
    if len(new) > 1:
        raise RuntimeError("Ambiguous concurrent launch; no windows will be closed")
    time.sleep(.1)
if run_dir is None:
    raise RuntimeError("VBS did not create a run directory")

user32 = ctypes.windll.user32
windows = []
owned = set()

@ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
def inspect_window(hwnd, unused):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value in owned:
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if rect.right - rect.left > 400 and rect.bottom - rect.top > 300:
            windows.append({"hwnd": hwnd, "pid": pid.value,
                            "visible": bool(user32.IsWindowVisible(hwnd)),
                            "minimized": bool(user32.IsIconic(hwnd)),
                            "rectangle": [rect.left, rect.top, rect.right, rect.bottom]})
    return True

for _ in range(300):
    owned = {p.pid for p in psutil.process_iter(["name", "cmdline"])
             if p.info["name"].lower() == "sumo-gui.exe" and
             any(str(run_dir).lower() in arg.lower() for arg in p.info["cmdline"] or [])}
    windows.clear()
    user32.EnumWindows(inspect_window, 0)
    if any(w["visible"] and not w["minimized"] for w in windows):
        break
    time.sleep(.1)
if not windows:
    raise RuntimeError("No native window found for the launched SUMO process")
progress = 0.0
for _ in range(600):
    events_file = run_dir / "events.jsonl"
    if events_file.is_file():
        lines = events_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["kind"] == "traffic_sample":
                progress = max(progress, event["sim_time_s"])
    if progress >= 5:
        break
    time.sleep(.1)
for window in windows:
    # The PID and exact unique output directory identified this task's own
    # process. Do not affect any pre-existing SUMO window or other application.
    user32.PostMessageW(window["hwnd"], 0x0010, 0, 0)  # WM_CLOSE
for _ in range(200):
    if (run_dir / "summary.json").is_file():
        break
    time.sleep(.1)
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
record = {"command": ["wscript.exe", str(ROOT / "Run SUMO Simulation.vbs")],
          "caller_working_directory": str(ROOT.parent.parent),
          "wscript_exit_code": launch.returncode, "windows": windows,
          "run_directory": str(run_dir), "summary": summary,
          "observed_traffic_sample_s_before_close": progress,
          "elapsed_wall_s": time.monotonic() - started,
          "closed_only_own_native_window": True}
record["passed"] = (launch.returncode == 0 and
                    any(w["visible"] and not w["minimized"] for w in windows) and
                    summary["status"] == "window_closed" and
                    summary["sim_time_s"] >= 2 and
                    summary["authored_files_unchanged"] and summary["sumo_exit_code"] == 0)
OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
raise SystemExit(0 if record["passed"] else 1)

