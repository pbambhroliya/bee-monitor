@echo off
rem Starts the bee labeller. First start: installs Streamlit into a local .venv folder.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo Installing, only the first time...
  python -m venv .venv || (echo Please install Python 3.10+ from python.org and try again. & pause & exit /b 1)
  ".venv\Scripts\pip" install -q -r requirements.txt || (echo Installation failed. & pause & exit /b 1)
)
".venv\Scripts\streamlit" run labeler\app.py
pause
