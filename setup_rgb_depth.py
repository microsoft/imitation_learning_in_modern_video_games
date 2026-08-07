#!/usr/bin/env python
"""Setup 4-channel (RGB+Depth) training data"""

import cv2
import h5py
import numpy as np
from pathlib import Path

def load_depth_video(video_path, target_size=(280, 150)):
    """Load depth video and resize to match game frames"""
    print(f"Loading depth video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    depth_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Resize to match game resolution
        gray = cv2.resize(gray, target_size)
        depth_frames.append(gray)

    cap.release()

    depth_frames = np.array(depth_frames, dtype=np.uint8)
    print(f"Loaded {len(depth_frames)} depth frames: {depth_frames.shape}")
    return depth_frames


def add_depth_to_hdf5(hdf5_path, depth_frames):
    """Add depth channel to HDF5 file"""
    print(f"Adding depth to {hdf5_path}...")

    with h5py.File(hdf5_path, 'r+') as f:
        num_frames = min(len(depth_frames), len([k for k in f.keys() if k.startswith('frame_')]))

        # Store depth frames
        for i in range(num_frames):
            if f'frame_{i}_depth' not in f:
                f.create_dataset(f'frame_{i}_depth', data=depth_frames[i], compression='gzip')

    print(f"Added {num_frames} depth frames")


def create_4channel_config():
    """Create config for 4-channel (RGB+Depth) training"""

    config = """seed: 42
checkpoint: null
resume_from: null

log:
  minimal_metrics: True
  checkpoint_epoch_freq: 100
  checkpoint_top_k_validation: 0
  checkpoint_last: True
  device_stats_monitor: True
  weights: False
  weights_step_freq: 100
  weights_log_scalars: False

trainer:
  profiler: simple
  max_steps: 500000
  gradient_clip_algorithm: norm
  gradient_clip_val: 1.0
  precision: 16-mixed

data:
  game: "csgo"
  data_path: C:\\csgo_data
  train_split_file_path: paper_data_split/csgo_dm_train_files.txt
  validation_split_file_path: null
  test_split_file_path: null
  image_height: 128
  image_width: 128
  framestacking: 1
  num_actions: 15
  num_joystick_actions: 4
  num_trigger_actions: 1
  num_button_actions: 10
  batch_size: 32
  sequence_length: 100
  downsample: 2
  prefetch_factor: 5
  train_num_workers: 0
  other_num_workers: 0
  decord_num_workers: 0

model:
  encoder:
    type: cnn
    cnn_encoder_dim: 4
    cnn_encoder_start_channels: 32
    mlp_encoder_hidden_size: 128
    mlp_encoder_num_layers: 3
    use_image_augmentation: False

  bc_policy:
    type: lstm
    hidden_size: 512
    num_layers: 4
    lstm_hidden_size: 512
    lstm_num_layers: 2
    gpt_num_layers: 4
    gpt_num_heads: 8
    gpt_embedding_dim: 512
    gpt_use_positional_encoding: False
    gpt_bias: True
    gpt_is_causal: True

  learning_rate: 3e-4
  lr_warmup_steps: 0
  joystick_loss_weight: 1.0
  button_loss_weight: 1.0
  train_from_embeddings: False
"""

    config_path = Path(r"C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games\pixelbc\configs\csgo\cnn_lstm_rgbd.yaml")
    with open(config_path, 'w') as f:
        f.write(config)

    print(f"\nCreated config: {config_path}")
    return config_path


if __name__ == '__main__':
    # Load depth video
    depth_video = Path(r"C:\workspace\NAVIGATION\Depth-Anything-V2\vis_video_depth\sequence_demo.mp4")

    if depth_video.exists():
        depth_frames = load_depth_video(str(depth_video))

        # Add depth to first HDF5 file as example
        hdf5_file = Path(r"C:\csgo_data\hdf5_dm_july2021_1.hdf5")
        if hdf5_file.exists():
            add_depth_to_hdf5(str(hdf5_file), depth_frames)

    # Create config
    config_path = create_4channel_config()

    print("\n" + "="*60)
    print("To train with RGB+Depth (4 channels):")
    print(f"python pixelbc/run_train.py {config_path} \\")
    print("  trainer.accelerator=cpu trainer.max_epochs=50")
    print("="*60)
