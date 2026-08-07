@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

REM Train with CNN encoder + LSTM policy (deathmatch files)
python pixelbc\run_train.py ^
    pixelbc\configs\csgo\cnn_lstm.yaml ^
    data.data_path=C:\csgo_data ^
    data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=50 %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\default_seed_42\
