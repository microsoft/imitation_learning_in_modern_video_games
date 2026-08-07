@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Checking for models...

set CKPT=%cd%\training_logs\cnn_lstm_rgbd_seed_42\models\last.ckpt
set PRETRAINED=%cd%\trained_models\ak47_sub_55k_drop_d4_dmexpert_28.h5

if exist "%CKPT%" (
    echo Using trained checkpoint
    set MODEL_PATH=!CKPT!
) else if exist "%PRETRAINED%" (
    echo Using pretrained model
    set MODEL_PATH=!PRETRAINED!
) else (
    echo ERROR: No model found
    echo Checked: %CKPT%
    echo Checked: %PRETRAINED%
    pause
    exit /b 1
)

echo Model: !MODEL_PATH!
echo Launching inference...
python pixelbc\online_rollout\csgo_rollout.py --checkpoint_path "!MODEL_PATH!" --save_dir rollout_videos --fps 16
echo Done
pause
