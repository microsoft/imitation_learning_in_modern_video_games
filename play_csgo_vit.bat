@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the end-to-end ViT 128 +Aug model (own_vit128_aug). Like
REM the ResNet bot: a from-scratch encoder trained on raw frames -- NOT a DINO
REM model -- so no DINO_SPATIAL_POOL is needed. The rollout resizes each captured
REM frame to the model's own image_shape (128x128) read from the checkpoint.
REM
REM Prereqs: CS:GO (legacy) on Dust II DM, AK-47, window FOCUSED;
REM   console: m_rawinput 0  +  sensitivity 2.5.
REM Usage:
REM   play_csgo_vit.bat                              default vit ckpt @ 16 fps
REM   play_csgo_vit.bat path\to\other.ckpt 8         custom ckpt + fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=training_logs\own_vit128_v2_seed_0\models\last.ckpt"
set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

if not exist "%CKPT%" (
  echo ERROR: vit checkpoint not found: %CKPT%
  echo   ^(run train_vit128_aug.bat first; it must save at least one epoch^)
  exit /b 1
)

echo ===================================================================
echo   ViT 128 +Aug bot (end-to-end transformer encoder)
echo   ckpt: %CKPT%
echo   fps:  %FPS%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" pixelbc\online_rollout\csgo_rgb_rollout.py --checkpoint "%CKPT%" --fps %FPS%
echo [EXIT] vit bot closed
endlocal
