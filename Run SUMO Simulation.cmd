@echo off
"%~dp0.venv\Scripts\python.exe" -B "%~dp0simulation\cooperative_loop_1km\run_simulation.py" %*
if errorlevel 1 pause
