@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM One-time precompute of frozen DINOv2-vitb14 embeddings from the DEPTH channel.
REM Feeds frame_i_depth (replicated 1->3ch) through the same frozen encoder, so the
REM RGB+D loader can concat [rgb|depth] -> 1536-d per frame.
REM Output: C:\csgo_data\embeddings\dino_depth\dinov2_vitb14\ .  Rerun-safe.

echo.
echo Precomputing DINOv2-vitb14 DEPTH embeddings (train+val+test)...
echo Output: C:\csgo_data\embeddings\dino_depth\dinov2_vitb14\
echo.

python pixelbc\scripts\generate_depth_embeddings.py --encoder_config pixelbc\configs\csgo\aim_expert_dino_vitb.yaml --data_base_path C:\csgo_data --filelist pixelbc\data\paper_data_split\csgo_dm_train_files.txt pixelbc\data\paper_data_split\csgo_dm_val_files.txt pixelbc\data\paper_data_split\csgo_dm_test_files.txt --output_dir C:\csgo_data --encoder_subdir dino_depth

echo.
echo Depth embeddings done. Next: wire the RGB+D loader/model, then train_dino_rgbd.bat.
