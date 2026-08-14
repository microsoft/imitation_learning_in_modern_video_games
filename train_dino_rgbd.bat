@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM RGB+D DINO: frozen DINOv2-vitb14 on RGB *and* depth, embeddings concatenated
REM (1536-d) -> LSTM. Clean-ablation twin of train_dino_rgb.bat: identical encoder,
REM LSTM, 299 split, epochs -- the ONLY difference is the added depth-embedding stream.
REM Requires BOTH precomputes done:
REM   precompute_dino_embeddings.bat        (RGB   -> embeddings/dino/dinov2_vitb14)
REM   precompute_dino_depth_embeddings.bat  (depth -> embeddings/dino_depth/dinov2_vitb14)

echo.
echo Training RGB+D DINOv2-vitb14 + LSTM (concat rgb+depth embeddings, 1536-d)...
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\aim_expert_dino_vitb_rgbd.yaml run_name=aim_expert_dino_vitb_rgbd_spatial data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt data.train_num_workers=0 data.other_num_workers=0 data.decord_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 log.checkpoint_epoch_freq=5 log.checkpoint_top_k_validation=3 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\aim_expert_dino_vitb_rgbd_seed_0\
