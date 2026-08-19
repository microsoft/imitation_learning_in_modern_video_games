@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM Same base setup as train_dino_rgb.bat (frozen DINOv2-vitb14 + LSTM, embeddings
REM mode, RGB-only encoder, no depth_embedding), PLUS the predict_depth auxiliary
REM task: RGBDecoder (output_channels=1) reconstructs the depth map in image space
REM from the LSTM's hidden state, logged as depth_loss alongside the usual
REM joystick/trigger/button metrics.
REM Requires make_depth.py to have already been run on the train/val split files
REM (frame_i_depth must exist in every file -- no graceful-fallback for missing depth
REM here, unlike require_depth_embedding).

echo.
echo Training DINOv2-vitb14 + LSTM + predict_depth auxiliary task...
echo First run downloads dinov2_vitb14 weights via torch.hub and precomputes embeddings.
echo.

REM run_name changed to _128: PREDICT_DEPTH_SIZE changed 64->128 (decoder output size,
REM raw-depth payload size), which changes checkpoint shapes -- resuming into the old
REM 64x64 run_name would crash. _128 starts a fresh run dir instead of touching the old one.
python pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb_predict_depth.yaml run_name=aim_expert_dino_vitb_predict_depth_128 data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=1 log.checkpoint_epoch_freq=1 log.checkpoint_top_k_validation=3 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\aim_expert_dino_vitb_predict_depth_128_seed_0\
