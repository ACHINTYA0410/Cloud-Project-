@echo off
REM ════════════════════════════════════════════════
REM  start_backend.bat — Launch the Flask backend
REM  Double-click this file or run from cmd/PowerShell
REM ════════════════════════════════════════════════

echo.
echo  =========================================
echo   Power Anomaly Detection — Flask Backend
echo  =========================================
echo.

REM Move into the backend folder (same dir as this script)
cd /d "%~dp0"

echo  [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo  ERROR: Python is not installed or not in PATH.
    echo  Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo.
echo  [2/3] Installing dependencies (from requirements.txt)...
pip install -r requirements.txt --quiet

echo.
echo  [3/3] Starting Flask server on http://localhost:5000
echo        Press Ctrl+C to stop.
echo.
python app.py

pause
