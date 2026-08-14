@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the SPATIAL RGB DINO model (12288-d patch tokens).
REM DINO_SPATIAL_POOL=4 builds the 12288-input encoder so the _spatial checkpoint
REM loads (POOL=0 in the baseline bat is for the OLD 768 CLS model and would
REM mismatch). Points at the _alex spatial run by default; the checkpoint is a
REM plain state_dict so loading it from this (-rgbd) rollout script is fine.
REM
REM Prereqs: CS:GO (legacy) on Dust II DM, AK-47, window FOCUSED;
REM   console: m_rawinput 0  +  sensitivity 2.5.
REM Usage:
REM   play_csgo_rgb_spatial.bat                              default spatial ckpt @ 16 fps
REM   play_csgo_rgb_spatial.bat path\to\other.ckpt 8         custom ckpt + fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

REM Spatial 12288-d encoder (must match how the checkpoint was trained).
set "DINO_SPATIAL_POOL=4"

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=..\imitation_learning_in_modern_video_games_alex\training_logs\aim_expert_dino_vitb_spatial_seed_0\models\last.ckpt"
set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

if not exist "%CKPT%" (
  echo ERROR: spatial checkpoint not found: %CKPT%
  echo   ^(the _spatial run must have saved at least one epoch first^)
  exit /b 1
)

echo ===================================================================
echo   RGB DINO bot -- SPATIAL 12288
echo   ckpt: %CKPT%
echo   fps:  %FPS%   DINO_SPATIAL_POOL=%DINO_SPATIAL_POOL%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" pixelbc\online_rollout\csgo_rgb_rollout.py --checkpoint "%CKPT%" --fps %FPS%
echo [EXIT] rgb spatial bot closed
endlocal
