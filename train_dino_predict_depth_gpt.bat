@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM GPT-policy variant of predict_depth (see train_dino_predict_depth.bat for the LSTM
REM version). Same frozen DINOv2-vitb14 encoder, RGBDecoder (output_channels=1)
REM reconstructs the depth map from the GPT policy's hidden state (gpt_out, pre-mlp_out
REM projection) instead of the LSTM's -- see pico_gpt.py's return_hidden support.
REM lr_warmup_steps=2000 is required (BCModel.__init__ asserts warmup>0 for GPT policies).
REM Requires make_depth.py to have already been run on the train/val split files.

echo.
echo Training DINOv2-vitb14 + GPT + predict_depth auxiliary task...
echo First run downloads dinov2_vitb14 weights via torch.hub and precomputes embeddings.
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb_predict_depth_gpt.yaml run_name=aim_expert_dino_vitb_predict_depth_gpt data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=1 log.checkpoint_epoch_freq=1 log.checkpoint_top_k_validation=3 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\aim_expert_dino_vitb_predict_depth_gpt_seed_0\
