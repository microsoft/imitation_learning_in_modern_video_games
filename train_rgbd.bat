@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Clear stray distributed env vars (LOCAL_RANK -> Lightning picks DDP/NCCL, absent on Windows)
set "LOCAL_RANK="
set "WORLD_SIZE="
set "RANK="
set "NODE_RANK="

REM cnn_lstm_rgbd.yaml is the big LSTM model, 3-channel RGB (it does NOT use depth,
REM despite the filename). LSTM head suits temporal actions like held movement keys.
REM Old checkpoints used the keys-dropped labels, so pass resume_from=scratch to
REM retrain on the fixed (keys-included) labels.

echo.
echo Training RGB LSTM model (GPU, auto-resume from latest checkpoint)...
echo.

python pixelbc\run_train.py pixelbc\configs\csgo\cnn_lstm_rgbd.yaml data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.train_num_workers=0 data.other_num_workers=0 trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 log.minimal_metrics=false %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\cnn_lstm_rgbd_seed_42\
