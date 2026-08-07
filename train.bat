@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

python pixelbc\run_train.py ^
    pixelbc\configs\csgo\default.yaml ^
    data.data_path=C:\csgo_data ^
    data.train_split_file_path=paper_data_split/csgo_train_files_small.txt ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=50 %1 %2 %3 %4 %5

echo.
echo Training complete. Results in: training_logs\default_seed_42\
