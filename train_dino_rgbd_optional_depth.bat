@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM Same as train_dino_rgbd.bat (frozen DINOv2-vitb14 on RGB+depth, concatenated
REM 1536-d embeddings -> LSTM), except data.require_depth_embedding=false: a file
REM missing its depth embedding falls back to zeros (depth off for that file, with
REM a warning) instead of crashing the whole run. Use this if not every clip has
REM had make_depth.py / generate_depth_embeddings.py run on it yet.

echo.
echo Training RGB+D DINOv2-vitb14 + LSTM (concat rgb+depth embeddings, 1536-d, depth optional per-file)...
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb_rgbd.yaml run_name=aim_expert_dino_vitb_rgbd_optional_depth data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 data.require_depth_embedding=false trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 log.checkpoint_epoch_freq=5 log.checkpoint_top_k_validation=3 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\aim_expert_dino_vitb_rgbd_optional_depth_seed_0\
