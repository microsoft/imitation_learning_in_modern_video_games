@echo off
REM Upload the trained models (checkpoints + TensorBoard logs, ~470 MB) to horde.
REM WSL rsync: resumable, skips already-uploaded files. Prompts for password once.
REM
REM Includes both LSTM arms:
REM   cnn_lstm_rgbd_seed_42  (depth 4-ch)
REM   cnn_lstm_rgb_seed_42   (no-depth 3-ch, the ablation twin)
REM Dest: horde (10.57.233.24) -> ~/csgo_training_logs/

echo Uploading trained models + logs (~470 MB) to horde via rsync...
echo (enter horde password when prompted)
echo.

wsl -d Ubuntu-24.04 rsync -avP /mnt/c/workspace/NAVIGATION/imitation_learning_in_modern_video_games-rgbd/training_logs/ horde@10.57.233.24:~/csgo_training_logs/

echo.
echo Done (or interrupted -- just rerun this .bat to resume).
