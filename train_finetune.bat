@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

REM Fine-tune pretrained model on CS:GO deathmatch data
python pixelbc\run_train.py ^
    pixelbc\configs\csgo\default.yaml ^
    data.data_path=C:\csgo_data ^
    data.train_split_file_path=paper_data_split/csgo_train_files_small.txt ^
    checkpoint=trained_models/ak47_sub_55k_drop_d4.h5 ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=10 ^
    model.learning_rate=1e-4 %1 %2 %3 %4 %5

echo.
echo Fine-tuning complete. Results in: training_logs\default_seed_42\
