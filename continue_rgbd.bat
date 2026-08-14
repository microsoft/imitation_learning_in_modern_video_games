@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Continue training the RGB+D DINO model (1536: 768 RGB (+) 768 depth) FROM the
REM existing epoch-44 checkpoint. Loads it via fit.ckpt_path (weights + optimizer +
REM epoch), validates every epoch, and saves a checkpoint every epoch.
REM
REM Prereqs (already done this session): 768 CLS embeddings regenerated for both
REM dino/ and dino_depth/ in C:\csgo_data; bc_model depth-doubling + data-loader
REM depth-concat restored.
REM
REM Usage:
REM   continue_rgbd.bat                       resume to epoch 60
REM   continue_rgbd.bat trainer.max_epochs=80 override any config key

REM Clear stray distributed env vars (Windows has no NCCL).
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

set "CKPT=C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games-rgbd\training_logs\aim_expert_dino_vitb_rgbd_seed_0\models\last.ckpt"
if not exist "%CKPT%" ( echo ERROR: checkpoint not found: %CKPT% & exit /b 1 )

echo Continuing RGB+D training from %CKPT% ...
.venv\Scripts\python.exe pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb_rgbd.yaml run_name=aim_expert_dino_vitb_rgbd data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=60 trainer.check_val_every_n_epoch=1 log.checkpoint_epoch_freq=1 log.checkpoint_top_k_validation=3 fit.ckpt_path=%CKPT% %1 %2 %3 %4 %5

echo.
echo Training ended. Results in: training_logs\aim_expert_dino_vitb_rgbd_seed_0\
endlocal
