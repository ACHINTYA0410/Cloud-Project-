@echo off
REM ════════════════════════════════════════════════
REM  run_training.bat — Train the Isolation Forest
REM  on the real UCI household power consumption CSV
REM ════════════════════════════════════════════════

echo.
echo  =========================================
echo   Isolation Forest Training (Real CSV)
echo  =========================================
echo.
echo  Dataset : ..\individual+household+electric+power+consumption\
echo  Rows    : Full Dataset (~2M rows, approx 10-15 minutes)
echo  Output  : isolation_forest.pkl + scaler.pkl + model_metadata.json
echo.

REM Move into the backend folder (same dir as this script)
cd /d "%~dp0"

echo  [1/2] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo  [2/2] Running train_model.py...
echo.
python train_model.py

echo.
echo  ─────────────────────────────────────────
if errorlevel 1 (
    echo  ERROR: Training failed. See output above.
) else (
    echo  SUCCESS! Model artifacts saved to backend\
    echo  Restart Flask backend to load the new model.
)
echo  ─────────────────────────────────────────
echo.
pause
