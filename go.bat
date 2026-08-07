@echo off
REM Complete CS:GO workflow

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo CS:GO Training Workflow
echo ============================================================
echo.

REM Step 1: Check/Download
if not exist C:\csgo_data (
    echo [1/3] Downloading CS:GO data...
    call download.bat
    if !errorlevel! neq 0 exit /b 1
) else (
    for /f %%i in ('dir /b C:\csgo_data\*.hdf5 2^>nul ^| find /c ".hdf5"') do set COUNT=%%i
    echo [1/3] Found !COUNT! files in C:\csgo_data
)

echo.
echo [2/3] Training...
call train.bat

echo.
echo [3/3] Complete
echo.
echo Results: training_logs\default_seed_42\
echo.
echo To evaluate:
echo   1. Launch CS:GO
echo   2. Load aim assist map: map aimtraindriving_v3 custom
echo   3. Run: rollout.bat csgo training_logs\default_seed_42\models\last.ckpt
echo.
