#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Testing TensorFlow model loading..."
python csgo_rollout_tensorflow.py --model_path trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5 --save_dir rollout_videos

echo "Done"
