@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Setup RGB+Depth data
python setup_rgb_depth.py

echo.
echo Training with 4-channel RGB+Depth input...
echo.

python pixelbc\run_train.py ^
    pixelbc\configs\csgo\cnn_lstm_rgbd.yaml ^
    data.data_path=C:\csgo_data ^
    data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=50

echo.
echo Training complete. Results in: training_logs\default_seed_42\
