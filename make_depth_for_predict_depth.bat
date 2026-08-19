@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM One-time precompute of frame_i_depth (Depth-Anything-V2, vits encoder -- do NOT
REM change it, see DATA_PREP.md: vits reproduces training depth to a 100% exact
REM match) for the train+val split files used by train_dino_predict_depth.bat.
REM Resumable: frames/files that already have frame_i_depth are skipped (e.g. if
REM this was already run for the RGB+D depth_embedding pipeline).
REM Runs on GPU.

echo.
echo Generating depth for TRAIN split...
python make_depth.py --split paper_data_split/csgo_dm_train_files.txt --data_dir C:\csgo_data

echo.
echo Generating depth for VAL split...
python make_depth.py --split paper_data_split/csgo_dm_val_files.txt --data_dir C:\csgo_data

echo.
echo Done. frame_i_depth is now present in C:\csgo_data\*.hdf5 for train+val files.
echo Next: train_dino_predict_depth.bat
