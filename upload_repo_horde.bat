@echo off
REM Upload the pixelbc repo (source + configs + bats, NOT .venv/data/logs -- those
REM have their own scripts: upload_csgo_horde.bat, upload_models_horde.bat) to
REM horde, so training scripts like train_dino_predict_embedding.bat can run there.
REM WSL rsync: resumable, skips already-uploaded files. Prompts for password once.
REM
REM Source : this repo (excludes .venv, .git, training_logs, __pycache__)
REM Dest   : horde (10.57.233.24) -> ~/imitation_learning_in_modern_video_games-rgbd/

echo Uploading pixelbc repo (source + configs + bats) to horde via rsync...
echo (enter horde password when prompted)
echo.

wsl -d Ubuntu-24.04 rsync -avP --exclude=.venv --exclude=.git --exclude=training_logs --exclude=__pycache__ /mnt/c/workspace/NAVIGATION/imitation_learning_in_modern_video_games-rgbd/ horde@10.57.233.24:~/imitation_learning_in_modern_video_games-rgbd/

echo.
echo Done (or interrupted -- just rerun this .bat to resume).
