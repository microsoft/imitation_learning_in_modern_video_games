@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM Held-out validation of a trained checkpoint.
REM Usage: validate.bat [config] [checkpoint]
REM   defaults: RGB model (default.yaml + last-v3.ckpt)
REM   RGBD:  validate.bat pixelbc\configs\csgo\cnn_lstm_rgbd.yaml training_logs\cnn_lstm_rgbd_seed_42\models\last.ckpt

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=pixelbc\configs\csgo\default.yaml"
set "CKPT=%~2"
if "%CKPT%"=="" set "CKPT=training_logs\default_seed_42\models\last-v3.ckpt"

python pixelbc\validate.py "%CONFIG%" checkpoint=%CKPT% data.data_path=C:\csgo_data data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt trainer.accelerator=gpu trainer.devices=1

echo.
echo Validation complete.
