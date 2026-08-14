@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Live CS:GO play with the original TeaPearce TensorFlow bot (stateful Keras-2).
REM The real fragger (offline aim corr ~0.6), unlike the DINO wanderer.
REM
REM Prereqs: CS:GO (legacy) running on a map, FOCUSED; AK-47 out.
REM   Play -> Deathmatch -> Dust II (training-matched), then run this.
REM
REM Usage:
REM   play_csgo_tf.bat                          dmexpert (deathmatch) @ 16 fps
REM   play_csgo_tf.bat aimexpertv2_60           swap in the aim-tuned model
REM   play_csgo_tf.bat dmexpert_28 8            model + custom fps

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

REM Keras-2 compat for the 2022 models (TF 2.21 = Keras 3 can't parse them).
set "TF_USE_LEGACY_KERAS=1"
set "TF_CPP_MIN_LOG_LEVEL=2"

REM Model tag (stem selector). Default = deathmatch expert.
set "TAG=%~1"
if "%TAG%"=="" set "TAG=dmexpert_28"
REM Accept EITHER a full model stem (e.g. July_remoterun7_g9_4k_n32_recipe_ton96__e14
REM or ak47_sub_55k_drop_d4) OR a short ak47 tag (dmexpert_28 / aimexpertv2_60).
if exist "trained_models\%TAG%_stateful.h5" (
  set "MODEL=trained_models\%TAG%_stateful"
) else if exist "trained_models\ak47_sub_55k_drop_d4_%TAG%_stateful.h5" (
  set "MODEL=trained_models\ak47_sub_55k_drop_d4_%TAG%_stateful"
) else (
  echo ERROR: no stateful model for '%TAG%'
  echo   full stems: July_remoterun7_g9_4k_n32_recipe_ton96__e14 ^| ak47_sub_55k_drop_d4
  echo   ak47 tags:  dmexpert_28 ^| aimexpertv2_60
  exit /b 1
)

set "FPS=%~2"
if "%FPS%"=="" set "FPS=16"

echo ===================================================================
echo   TF CS:GO bot (CPU, ~14.6 fps single-frame)
echo   model: %MODEL%
echo   fps:   %FPS%
echo   Focus CS:GO (Dust II DM, AK-47). Ctrl-C to stop.
echo ===================================================================
"%PYEXE%" csgo_rollout_tf_live.py --model "%MODEL%" --fps %FPS%
echo [EXIT] tf bot closed
endlocal
