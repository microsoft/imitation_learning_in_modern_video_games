@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM One-time precompute of frozen DINOv2-vitb14 embeddings for the CSGO data.
REM Required before train_dino_rgb.bat (which uses train_from_embeddings=True).
REM Runs the frozen encoder over every frame of train+val+test splits and caches
REM .npz files under C:\csgo_data\embeddings\dino\dinov2_vitb14\ .
REM Rerun-safe. If it OOMs, add:  --batch_size 512  (or lower) to the line below.

echo.
echo Precomputing DINOv2-vitb14 embeddings for CSGO (train+val+test)...
echo Output: C:\csgo_data\embeddings\dino\dinov2_vitb14\
echo.

python pixelbc\scripts\generate_embeddings.py --game csgo --encoder_config pixelbc\configs\csgo\aim_expert_dino_vitb.yaml --data_base_path C:\csgo_data --filelist pixelbc\data\paper_data_split\csgo_dm_train_files.txt pixelbc\data\paper_data_split\csgo_dm_val_files.txt pixelbc\data\paper_data_split\csgo_dm_test_files.txt --output_dir C:\csgo_data

echo.
echo Embeddings done. Now run train_dino_rgb.bat.
