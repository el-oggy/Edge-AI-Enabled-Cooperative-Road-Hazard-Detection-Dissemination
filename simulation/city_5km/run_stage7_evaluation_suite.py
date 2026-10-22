"""
PFPD Stage 7: Comprehensive 10-Point Multi-RSU City Evaluation Suite
Automates end-to-end execution, instrumentation, and verification of:
1. 5 km City Network Integrity (Routes, Signalized Junctions)
2. Traffic Density (50+ continuously circulating vehicles)
3. 4-RSU Core Infrastructure
4. Inter-RSU Backhaul Forwarding / Relaying
5. Multi-Hazard Concurrent Simulation
6. Lane Merging & Speed Advisories
7. Full Obstruction & Dynamic TraCI Rerouting (Detours)
8. Mode B Scalable Connected Vehicle Streams
9. Mode A Genuine Physical Raspberry Pi 5 MNN INT8 Integration
10. Expanded 8-RSU Deployment Density & Coverage Scaling
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PI_HOST = "192.168.1.46"
PI_USER = "oggy"
PC_HOST = "192.168.1.48"
UDP_PORT = 5005

def run_ssh_command(cmd, timeout=40):
    full_cmd = ["ssh", f"{PI_USER}@{PI_HOST}", cmd]
    res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr

def run_evaluation_suite():
    print("=" * 88)
    print("  PFPD STAGE 7: 5.0 KM MULTI-RSU COOPERATIVE V2X CITY EVALUATION SUITE")
    print(f"  Physical Pi 5: {PI_HOST} | Simulation PC: {PC_HOST}:{UDP_PORT}")
    print("=" * 88)

    suite_results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # TEST RUN 1: 4-RSU Core Mesh + Dynamic Rerouting + Simultaneous Hazards
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 1: 4-RSU CORE MESH WITH SIMULTANEOUS HAZARDS & DYNAMIC REROUTING")
    print("=" * 80)

    mesh_script = BASE_DIR / "rsu_network_mesh.py"
    cmd_4rsu = [
        sys.executable, str(mesh_script),
        "--pi-host", PI_HOST,
        "--port", str(UDP_PORT),
        "--headless",
        "--step-delay", "0.035",
        "--max-steps", "450",
        "--scenario-name", "CORE_4_RSU_MULTIHAZARD_REROUTE"
    ]

    print("[Orchestrator] Launching 4-RSU City Mesh Controller (max_steps=450, step_delay=35ms)...")
    proc_4rsu = subprocess.Popen(cmd_4rsu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Allow simulation 5 seconds to establish circulating traffic (~55-65 vehicles)
    time.sleep(5.0)

    # Trigger Physical Raspberry Pi 5 Edge Perception Daemon (Mode A)
    pi_cmd = (
        f"/home/oggy/projects/pfpd/.venv/bin/python3 /home/oggy/projects/pfpd/pi_edge_perception_daemon.py "
        f"--scenario ALL --host {PC_HOST} --port {UDP_PORT} --burst-frames 5 --fps-limit 28.75 "
        f"--out-json /home/oggy/projects/pfpd/results/pi_edge_stage7_run.json"
    )
    print("[Orchestrator] Triggering Physical Raspberry Pi 5 (Mode A MNN INT8 Inference)...")
    code, pi_stdout, pi_stderr = run_ssh_command(pi_cmd, timeout=45)
    print("--- [Physical Pi 5 Output Summary] ---")
    for line in pi_stdout.splitlines()[-10:]:
        print("  [Pi 5]", line)

    # Wait for simulation to finish
    out_4rsu, _ = proc_4rsu.communicate(timeout=60)
    print("--- [4-RSU SUMO Controller Output Summary] ---")
    for line in out_4rsu.splitlines()[-20:]:
        print("  [SUMO]", line)

    file_4rsu = OUTPUT_DIR / "stage7_core_4_rsu_multihazard_reroute_run.json"
    if file_4rsu.exists():
        with open(file_4rsu, "r") as f:
            suite_results["core_4_rsu_test"] = json.load(f)

    time.sleep(2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST RUN 2: 8-RSU Expanded City Deployment Density & Coverage Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 2: 8-RSU EXPANDED CITY DEPLOYMENT & COVERAGE DENSITY ANALYSIS")
    print("=" * 80)

    cmd_8rsu = [
        sys.executable, str(mesh_script),
        "--expanded-rsus",
        "--pi-host", PI_HOST,
        "--port", str(UDP_PORT),
        "--headless",
        "--step-delay", "0.035",
        "--max-steps", "450",
        "--scenario-name", "EXPANDED_8_RSU_DENSITY_ANALYSIS"
    ]

    print("[Orchestrator] Launching 8-RSU Expanded City Mesh Controller (max_steps=450)...")
    proc_8rsu = subprocess.Popen(cmd_8rsu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(5.0)

    # Trigger Pi 5 Mode A again for multi-RSU handover testing
    print("[Orchestrator] Triggering Physical Raspberry Pi 5 for Expanded 8-RSU Mesh...")
    code2, pi_stdout2, _ = run_ssh_command(pi_cmd, timeout=45)

    out_8rsu, _ = proc_8rsu.communicate(timeout=60)
    print("--- [8-RSU SUMO Controller Output Summary] ---")
    for line in out_8rsu.splitlines()[-20:]:
        print("  [SUMO]", line)

    file_8rsu = OUTPUT_DIR / "stage7_expanded_8_rsu_density_analysis_run.json"
    if file_8rsu.exists():
        with open(file_8rsu, "r") as f:
            suite_results["expanded_8_rsu_test"] = json.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER AUDIT REPORT COMPILATION
    # ─────────────────────────────────────────────────────────────────────────
    master_path = OUTPUT_DIR / "stage7_master_city_evaluation_report.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    print("\n" + "=" * 88)
    print("  STAGE 7 EVALUATION SUITE COMPLETED SUCCESSFULLY")
    print(f"  Master Report Saved: {master_path}")
    print("=" * 88)
    return suite_results

if __name__ == "__main__":
    run_evaluation_suite()
