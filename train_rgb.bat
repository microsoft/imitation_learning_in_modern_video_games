@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM cnn_lstm_rgb.yaml = the NO-DEPTH LSTM arm (3-channel RGB). It is the clean-ablation
REM twin of cnn_lstm_rgbd.yaml: identical LSTM, identical 299-file split, identical
REM dropout -- the ONLY difference is use_depth (false here, true in the rgbd arm).
REM Logs to training_logs\cnn_lstm_rgb_seed_42\ (separate from the depth arm).

echo.
echo Training NO-DEPTH RGB LSTM model (GPU, auto-resume from latest checkpoint)...
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\cnn_lstm_rgb.yaml data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.train_num_workers=0 data.other_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\cnn_lstm_rgb_seed_42\
