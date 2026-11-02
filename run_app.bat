@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo First create the environment and install requirements:
  echo python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
  pause
  exit /b 1
)
.venv\Scripts\python.exe backend\app.py
pause
