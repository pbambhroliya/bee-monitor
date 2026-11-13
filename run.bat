@echo off
rem Start the bee monitoring app (Windows): double-click this file.
rem The first start installs everything into the folder .venv (5-15 minutes, needs internet).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First start: installing, 5-15 minutes...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  ".venv\Scripts\python" -m pip install --upgrade pip
  ".venv\Scripts\python" -m pip install -r requirements.txt
  ".venv\Scripts\python" -m pip install -e . --no-deps
)
".venv\Scripts\python" -m streamlit run app.py
pause
