@echo off
REM Train CS:GO aiming agent on small dataset (10 expert files)
REM Prerequisites:
REM   1. Download 10 CS:GO files using: download_csgo_small.bat
REM   2. Place in: C:\csgo_data\
REM   3. Run this script

setlocal enableextensions enabledelayedexpansion

set DATA_PATH=C:\csgo_data
set SPLIT_FILE=pixelbc\data\paper_data_split\csgo_train_files_small.txt
set EPOCHS=%1
if "!EPOCHS!"=="" set EPOCHS=50

echo ============================================================
echo CS:GO Small Dataset Training
echo ============================================================
echo Data Path: !DATA_PATH!
echo Split File: !SPLIT_FILE!
echo Epochs: !EPOCHS!
echo.

REM Check if data exists
if not exist "!DATA_PATH!" (
    echo ERROR: Data directory not found: !DATA_PATH!
    echo Please run: download_csgo_small.bat
    exit /b 1
)

REM Count downloaded files
for /f %%i in ('dir /b "!DATA_PATH!\*.hdf5" 2^>nul ^| find /c ".hdf5"') do set FILE_COUNT=%%i

if "!FILE_COUNT!"=="" (
    echo ERROR: No .hdf5 files found in !DATA_PATH!
    echo Please run: download_csgo_small.bat
    exit /b 1
)

echo Found: !FILE_COUNT! .hdf5 files
echo.

REM Activate venv
call .venv\Scripts\activate.bat

echo ============================================================
echo Starting Training
echo ============================================================
echo.
echo Expected results:
echo   - Success metric: Kills-per-minute (KPM)
echo   - Good result: 5+ KPM
echo   - Excellent result: 8+ KPM
echo.
echo Training logs: training_logs\default_seed_42\
echo Checkpoints: training_logs\default_seed_42\models\
echo.

python pixelbc\run_train.py ^
    pixelbc\configs\csgo\default.yaml ^
    data.data_path="!DATA_PATH!" ^
    data.train_split_file_path="!SPLIT_FILE!" ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=!EPOCHS! ^
    %2 %3 %4 %5 %6 %7 %8 %9

if !errorlevel! equ 0 (
    echo.
    echo ============================================================
    echo Training Complete!
    echo ============================================================
    echo.
    echo Next: Evaluate the trained model
    echo.
    echo 1. Launch CS:GO and load aim assist map:
    echo    In CS:GO console: map aimtraindriving_v3 custom
    echo.
    echo 2. Run evaluation:
    echo    rollout.bat csgo training_logs\default_seed_42\models\last.ckpt --path csgo_results\
    echo.
    echo 3. Check results:
    echo    python pixelbc\plotting\csgo_report_results.py csgo_results\
    echo.
) else (
    echo.
    echo ============================================================
    echo Training Failed
    echo ============================================================
    echo Check the error messages above
)
