@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM Paper's data-efficient setup: FROZEN DINOv2-vitb14 encoder + LSTM head,
REM train_from_embeddings=True (exactly aim_expert_dino_vitb.yaml). This is the
REM README/paper recommendation for a small dataset -- a pretrained backbone the
REM policy learns on top of, instead of a from-scratch CNN that collapses.
REM Only deviations from the paper config: our data path + 299 split, validation
REM split enabled, and num_workers=0 (Windows worker-spawn re-imports TF -> crash).

echo.
echo Training DINOv2-vitb14 + LSTM (frozen encoder, embeddings)...
echo First run downloads dinov2_vitb14 weights via torch.hub and precomputes embeddings.
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb.yaml run_name=aim_expert_dino_vitb_spatial data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 log.checkpoint_epoch_freq=5 log.checkpoint_top_k_validation=3 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\aim_expert_dino_vitb_seed_0\
