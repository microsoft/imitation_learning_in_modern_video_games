@echo off
REM Complete CS:GO workflow: download → train → evaluate
REM This script orchestrates the entire pipeline

setlocal enableextensions enabledelayedexpansion

set DATA_PATH=C:\csgo_data
set SPLIT_FILE=pixelbc\data\paper_data_split\csgo_train_files_small.txt
set LOG=csgo_workflow.log

cls
echo ============================================================
echo CS:GO Complete Workflow
echo ============================================================
echo This script will:
echo   1. Check/download CS:GO data
echo   2. Create split file
echo   3. Train agent
echo   4. Guide evaluation
echo.
echo Log: !LOG!
echo ============================================================
echo.

REM ============================================================
REM STEP 1: Check data
REM ============================================================
echo [STEP 1] Checking CS:GO data...
echo.

if exist "!DATA_PATH!" (
    for /f %%i in ('dir /b "!DATA_PATH!\*.hdf5" 2^>nul ^| find /c ".hdf5"') do set FILE_COUNT=%%i
    if !FILE_COUNT! geq 10 (
        echo [OK] Found !FILE_COUNT! .hdf5 files in !DATA_PATH!
    ) else (
        echo [WARNING] Found only !FILE_COUNT! files, need 10+
        echo.
        echo To download:
        echo   1. Visit: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#datasets
        echo   2. Get download URLs
        echo   3. Run: download_csgo_small.bat
        echo   4. Then retry: csgo_complete.bat
        echo.
        pause
        exit /b 1
    )
) else (
    echo [MISSING] Data directory not found: !DATA_PATH!
    echo.
    echo To download:
    echo   1. Visit: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#datasets
    echo   2. Get download URLs
    echo   3. Run: download_csgo_small.bat
    echo   4. Then retry: csgo_complete.bat
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [STEP 2] Verifying split file...
echo.

if exist "!SPLIT_FILE!" (
    echo [OK] Split file exists: !SPLIT_FILE!
) else (
    echo [ERROR] Split file not found: !SPLIT_FILE!
    exit /b 1
)

echo.
echo ============================================================
echo [STEP 3] Starting Training
echo ============================================================
echo.
echo Data: !DATA_PATH! (!FILE_COUNT! files)
echo Config: pixelbc\configs\csgo\default.yaml
echo Split: !SPLIT_FILE!
echo.
echo Expected:
echo   - Training time: 2-5 GPU hours (CPU: 8-10 hours)
echo   - Success metric: Kills-per-minute (5-8 KPM = good)
echo   - Checkpoints saved to: training_logs\default_seed_42\models\
echo.

pause
echo.
echo Starting training...
echo [%date% %time%] Training started >> "!LOG!"

call .venv\Scripts\activate.bat

python pixelbc\run_train.py ^
    pixelbc\configs\csgo\default.yaml ^
    data.data_path="!DATA_PATH!" ^
    data.train_split_file_path="!SPLIT_FILE!" ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=50

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Training failed
    echo Check the error messages above
    echo [%date% %time%] Training failed >> "!LOG!"
    pause
    exit /b 1
)

echo.
echo [%date% %time%] Training complete >> "!LOG!"
echo ============================================================
echo Training Complete!
echo ============================================================
echo.

REM ============================================================
REM STEP 4: Evaluation guide
REM ============================================================
echo.
echo ============================================================
echo [STEP 4] Ready to Evaluate
echo ============================================================
echo.
echo To evaluate the trained agent:
echo.
echo 1. LAUNCH CS:GO
echo    - Start the game
echo    - Load the custom aim assist map via console:
echo      map aimtraindriving_v3 custom
echo.
echo 2. RUN EVALUATION (in a separate terminal)
echo    rollout.bat csgo training_logs\default_seed_42\models\last.ckpt --path csgo_results\
echo.
echo 3. ANALYZE RESULTS
echo    python pixelbc\plotting\csgo_report_results.py csgo_results\
echo.
echo Expected metrics:
echo   - Kills-per-minute (KPM): 5-8 is good
echo   - Consistency: Low variance across 3 runs
echo   - Success rate: % of rollouts that hit target
echo.

echo ============================================================
echo Quick Commands
echo ============================================================
echo.
echo Evaluate now:
echo   rollout.bat csgo training_logs\default_seed_42\models\last.ckpt --path csgo_results\
echo.
echo Analyze results:
echo   python pixelbc\plotting\csgo_report_results.py csgo_results\
echo.
echo View training logs:
echo   tensorboard --logdir training_logs\default_seed_42\tb
echo.
echo Train longer for better results:
echo   train.bat csgo default.yaml data.data_path="!DATA_PATH!" data.train_split_file_path="!SPLIT_FILE!" trainer.max_epochs=100
echo.

echo [%date% %time%] Workflow complete >> "!LOG!"
echo.
echo Workflow log saved to: !LOG!
echo.
pause
