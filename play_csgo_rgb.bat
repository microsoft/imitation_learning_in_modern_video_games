@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the RGB DINO model (PyTorch). Baseline: wanders (moves)
REM but does NOT aim -- the CLS-token model. DINO_SPATIAL_POOL=0 forces the OLD
REM 768-d CLS encoder so the existing 768-d checkpoint loads. (The spatial 12288-d
REM aim fix is the default encoder now, but needs a retrain before it can be played.)
REM
REM Prereqs: CS:GO (legacy) on Dust II DM, AK-47, FOCUSED;
REM   console: m_rawinput 0  +  sensitivity 2.5  (stops spinning).
REM Usage:
REM   play_csgo_rgb.bat                                  default ckpt @ 16 fps
REM   play_csgo_rgb.bat training_logs\...\last.ckpt 8    custom ckpt + fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

REM CLS 768-d encoder so the existing checkpoint loads. Do NOT unset for this model.
set "DINO_SPATIAL_POOL=0"

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=training_logs\aim_expert_dino_vitb_seed_0\models\last.ckpt"
set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

echo ===================================================================
echo   RGB DINO bot (CLS 768 baseline -- wanders, weak/no aim)
echo   ckpt: %CKPT%
echo   fps:  %FPS%   DINO_SPATIAL_POOL=%DINO_SPATIAL_POOL%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" pixelbc\online_rollout\csgo_rgb_rollout.py --checkpoint "%CKPT%" --fps %FPS%
echo [EXIT] rgb bot closed
endlocal
