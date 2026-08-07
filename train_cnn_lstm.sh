#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python pixelbc/run_train.py pixelbc/configs/csgo/cnn_lstm.yaml data.data_path=C:/csgo_data trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=50
