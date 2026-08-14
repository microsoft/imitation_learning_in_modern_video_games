@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Train an end-to-end Vision Transformer (ViT) encoder + LSTM policy on raw
REM 128x128 CS:GO frames (config aim_expert_own_vit_custom_128.yaml,
REM train_from_embeddings=False). The transformer analog of the ResNet 128 run --
REM reads frames straight from the hdf5, no DINO embeddings involved.
REM
REM Usage:
REM   train_vit128_aug.bat                        defaults (20 epochs, val every 1)
REM   train_vit128_aug.bat trainer.max_epochs=40  override any config key

REM Clear stray distributed env vars (Windows has no NCCL).
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

echo Training end-to-end ViT 128 +Aug (own_vit128_aug) on C:\csgo_data ...
.venv\Scripts\python.exe pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_own_vit_custom_128.yaml run_name=own_vit128_aug data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=20 trainer.check_val_every_n_epoch=1 log.checkpoint_epoch_freq=1 log.checkpoint_top_k_validation=3 %1 %2 %3 %4 %5

echo.
echo Training ended. Results in: training_logs\own_vit128_aug_seed_0\
endlocal
