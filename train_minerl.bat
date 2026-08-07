@echo off
call .venv\Scripts\activate.bat

echo Training on MineRL dataset (already downloaded)...
echo.

python pixelbc\run_train.py ^
    pixelbc\configs\minerl\default.yaml ^
    trainer.accelerator=cpu ^
    trainer.max_epochs=50 %1 %2 %3 %4 %5

echo.
echo Training complete! Results in: training_logs\default_seed_42\
