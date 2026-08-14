@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the SPATIAL RGB+D DINO model (24576-d = 12288 RGB (+) 12288
REM depth). Each frame: capture -> Depth-Anything-V2 (vits) depth map -> DINO on both
REM RGB and depth -> concat -> policy. DINO_SPATIAL_POOL=4 makes each stream 12288 so
REM the _rgbd_spatial checkpoint loads (the depth rollout script's "1536" docstring is
REM the OLD 768+768 CLS wording; the concat is dim-agnostic).
REM
REM Prereqs: CS:GO (legacy) on Dust II DM, AK-47, window FOCUSED;
REM   console: m_rawinput 0  +  sensitivity 2.5.
REM Usage:
REM   play_csgo_rgbd_spatial.bat                             default rgbd spatial ckpt @ 16 fps
REM   play_csgo_rgbd_spatial.bat path\to\other.ckpt 8        custom ckpt + fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

REM Spatial 12288-d per stream (must match how the checkpoint was trained).
set "DINO_SPATIAL_POOL=4"

set "DEPTH_DIR=C:\workspace\NAVIGATION\Depth-Anything-V2"

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=..\imitation_learning_in_modern_video_games_alex\training_logs\aim_expert_dino_vitb_rgbd_spatial_seed_0\models\last.ckpt"
set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

if not exist "%CKPT%" (
  echo ERROR: rgbd spatial checkpoint not found: %CKPT%
  echo   ^(run train_dino_rgbd.bat first; it must save at least one epoch^)
  exit /b 1
)

echo ===================================================================
echo   RGB+D DINO bot -- SPATIAL 24576  (RGB (+) depth)
echo   ckpt:  %CKPT%
echo   depth: %DEPTH_DIR%  (Depth-Anything-V2 vits, live per frame)
echo   fps:   %FPS%   DINO_SPATIAL_POOL=%DINO_SPATIAL_POOL%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" pixelbc\online_rollout\csgo_depth_rollout.py --checkpoint "%CKPT%" --fps %FPS% --depth_anything_dir "%DEPTH_DIR%"
echo [EXIT] rgbd spatial bot closed
endlocal
