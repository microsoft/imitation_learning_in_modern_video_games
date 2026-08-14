@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the CURRENT (already-trained) RGB+D DINO model -- the
REM 768-d CLS version (1536-d = 768 RGB (+) 768 depth). DINO_SPATIAL_POOL=0 forces
REM the CLS encoder so this existing checkpoint loads. Each frame: capture ->
REM Depth-Anything-V2 (vits) depth map -> DINO on RGB and depth -> concat -> policy.
REM
REM (For the newer 12288-per-stream retrain use play_csgo_rgbd_spatial.bat instead.)
REM
REM Prereqs: CS:GO (legacy) on Dust II DM, AK-47, window FOCUSED;
REM   console: m_rawinput 0  +  sensitivity 2.5.
REM Usage:
REM   play_csgo_rgbd.bat                             default rgbd ckpt @ 16 fps (kills ON)
REM   play_csgo_rgbd.bat nokills                     disable GSI kill tracking
REM   play_csgo_rgbd.bat path\to\other.ckpt 8        custom ckpt + fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

REM CLS 768-d per stream so the existing rgbd checkpoint loads. Do NOT unset.
set "DINO_SPATIAL_POOL=0"

set "DEPTH_DIR=C:\workspace\NAVIGATION\Depth-Anything-V2"

REM Kill tracking is ON by default (GSI poll is non-blocking). First token
REM "nokills" turns it off; "kills" is still accepted as an explicit on. Either
REM keyword is shifted away so CKPT/FPS still read from the remaining args.
set "TRACK=--track_kills"
if /I "%~1"=="nokills" ( set "TRACK=" & shift )
if /I "%~1"=="kills" ( set "TRACK=--track_kills" & shift )

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=training_logs\aim_expert_dino_vitb_rgbd_seed_0\models\last.ckpt"
set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

if not exist "%CKPT%" (
  echo ERROR: rgbd checkpoint not found: %CKPT%
  exit /b 1
)

echo ===================================================================
echo   RGB+D DINO bot -- CLS 1536  (RGB (+) depth, current model)
echo   ckpt:  %CKPT%
echo   depth: %DEPTH_DIR%  (Depth-Anything-V2 vits, live per frame)
echo   fps:   %FPS%   DINO_SPATIAL_POOL=%DINO_SPATIAL_POOL%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" pixelbc\online_rollout\csgo_depth_rollout.py --checkpoint "%CKPT%" --fps %FPS% --depth_anything_dir "%DEPTH_DIR%" %TRACK%
echo [EXIT] rgbd bot closed
endlocal
