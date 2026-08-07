#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Checking for models..."

if [ -f "training_logs/cnn_lstm_rgbd_seed_42/models/last.ckpt" ]; then
    echo "Using trained checkpoint"
    MODEL_PATH="training_logs/cnn_lstm_rgbd_seed_42/models/last.ckpt"
elif [ -f "trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5" ]; then
    echo "Using pretrained model"
    MODEL_PATH="trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5"
else
    echo "ERROR: No model found"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo "Launching inference..."
python pixelbc/online_rollout/csgo_rollout.py --checkpoint_path "$MODEL_PATH" --save_dir rollout_videos --fps 16
echo "Done"
