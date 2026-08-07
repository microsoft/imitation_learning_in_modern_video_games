#!/usr/bin/env python
"""Train with grayscale depth as auxiliary input"""

import h5py
import cv2
import numpy as np
from pathlib import Path

def load_depth_frames(depth_video_path, num_frames=100):
    """Load grayscale depth frames from video"""
    print(f"Loading depth video: {depth_video_path}")

    cap = cv2.VideoCapture(depth_video_path)
    depth_frames = []

    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        depth_frames.append(gray)
        frame_count += 1

    cap.release()

    if depth_frames:
        depth_frames = np.array(depth_frames, dtype=np.float32) / 255.0
        print(f"Loaded {len(depth_frames)} depth frames: {depth_frames.shape}")
        return depth_frames
    return None


def augment_hdf5_with_depth(hdf5_file, depth_frames):
    """Add depth frames to HDF5 file as auxiliary data"""
    print(f"Augmenting {hdf5_file} with depth data...")

    with h5py.File(hdf5_file, 'r+') as f:
        # Create depth group
        if 'depth' in f:
            del f['depth']

        depth_group = f.create_group('depth')

        for i, depth_frame in enumerate(depth_frames):
            depth_group.create_dataset(f'frame_{i}', data=depth_frame, compression='gzip')

        print(f"Added {len(depth_frames)} depth frames to {hdf5_file}")


def create_multimodal_config():
    """Create training config for RGB + Depth input"""

    config = """
seed: 42
checkpoint: trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5
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

  # Depth encoder (separate)
  depth_encoder:
    enabled: True
    type: cnn
    cnn_encoder_dim: 2
    cnn_encoder_start_channels: 16
    mlp_encoder_hidden_size: 64
    mlp_encoder_num_layers: 2

  bc_policy:
    type: lstm
    hidden_size: 512
    num_layers: 4
    lstm_hidden_size: 512
    lstm_num_layers: 2

  learning_rate: 1e-4
  lr_warmup_steps: 0
  joystick_loss_weight: 1.0
  button_loss_weight: 1.0
  train_from_embeddings: False
"""

    config_path = Path(r"C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games\pixelbc\configs\csgo\cnn_lstm_depth.yaml")
    with open(config_path, 'w') as f:
        f.write(config)

    print(f"Created config: {config_path}")
    return config_path


if __name__ == '__main__':
    # Load depth video
    depth_video = Path(r"C:\workspace\NAVIGATION\Depth-Anything-V2\vis_video_depth\sequence_demo.mp4")

    if depth_video.exists():
        depth_frames = load_depth_frames(str(depth_video), num_frames=300)

        if depth_frames is not None:
            # Augment first HDF5 file with depth
            hdf5_file = Path(r"C:\csgo_data\hdf5_dm_july2021_1.hdf5")
            if hdf5_file.exists():
                augment_hdf5_with_depth(str(hdf5_file), depth_frames)

    # Create multimodal config
    config_path = create_multimodal_config()

    print("\n" + "="*60)
    print("To train with depth:")
    print(f"python pixelbc/run_train.py {config_path}")
    print("="*60)
