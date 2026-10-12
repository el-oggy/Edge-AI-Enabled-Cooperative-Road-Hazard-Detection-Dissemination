@echo off
REM Launch interactive SUMO-GUI with real-time TraCI V2X Dissemination
echo ==================================================================
echo Launching Edge-AI Cooperative Hazard Simulation in SUMO-GUI
echo Network: 1.0 km Signalized Corridor
echo ==================================================================

uv run --python 3.11 "%~dp0run_cooperative_v2x_sim.py"
pause
