@echo off
setlocal enableextensions enabledelayedexpansion

echo Starting CS:GO inference with trained model...
echo.
echo Requirements:
echo - CS:GO must be running
echo - Game window must be focused
echo - Model checkpoint must exist at: training_logs\cnn_lstm_rgbd_seed_42\models\last.ckpt
echo.

cd /d "%~dp0"

call .venv\Scripts\activate.bat

echo Launching inference...
python pixelbc\online_rollout\csgo_rollout.py ^
  --checkpoint_path training_logs\cnn_lstm_rgbd_seed_42\models\last.ckpt ^
  --save_dir rollout_videos ^
  --fps 16

echo.
echo Inference complete. Videos saved to: rollout_videos\
pause
