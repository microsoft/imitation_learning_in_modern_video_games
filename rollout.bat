@echo off
REM Evaluate a trained behaviour cloning agent in CS:GO or Minecraft
REM Requires: game to be running (CS:GO needs aim assist map loaded)
REM Usage:
REM   rollout.bat csgo path\to\checkpoint.ckpt
REM   rollout.bat csgo path\to\checkpoint.ckpt --fps 8 --path custom_results
REM   rollout.bat minerl path\to\checkpoint.ckpt

setlocal enableextensions enabledelayedexpansion

if "%1"=="" (
    echo Usage: rollout.bat ^<game^> ^<checkpoint^> [options]
    echo.
    echo game:       csgo or minerl
    echo checkpoint: path to model.ckpt file
    echo options:    --fps 16, --path results/, etc.
    exit /b 1
)

set GAME=%1
set CHECKPOINT=%2

if "!GAME!"=="" goto :usage
if "!CHECKPOINT!"=="" goto :usage

if /i "!GAME!"=="csgo" goto :game_ok
if /i "!GAME!"=="minerl" goto :game_ok
echo ERROR: game must be 'csgo' or 'minerl'
exit /b 1

:game_ok
if not exist "!CHECKPOINT!" (
    echo ERROR: checkpoint not found: !CHECKPOINT!
    exit /b 1
)

call .venv\Scripts\activate.bat

echo ============================================================
echo Rollout evaluation: !GAME!
echo Checkpoint: !CHECKPOINT!
echo ============================================================
echo.

if /i "!GAME!"=="csgo" (
    echo NOTE: Make sure CS:GO is running with the aim assist map loaded
    echo   map aimtraindriving_v3 custom
    echo.
    python pixelbc\online_rollout\rollout.py --game csgo --checkpoint "!CHECKPOINT!" %3 %4 %5 %6 %7 %8 %9
) else (
    python pixelbc\scripts\run_minerl_rollouts.py "!CHECKPOINT!" --output_dir results/ %3 %4 %5 %6 %7 %8 %9
)
