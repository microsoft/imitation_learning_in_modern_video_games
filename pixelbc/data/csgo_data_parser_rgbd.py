#!/usr/bin/env python
"""CSGO data parser with RGB+Depth (4-channel) support"""

import h5py
import numpy as np
from pathlib import Path

class CSGODataParserRGBD:
    """Load CSGO data as 4-channel (RGB+Depth) instead of 3-channel RGB"""

    def __init__(self, data_config):
        self.data_config = data_config

    def get_all_data(self, filename):
        """Load RGB frames and depth frames, concatenate to 4-channel"""
        print(f"Loading with depth: {filename}")

        data_path = Path(self.data_config.data_path) / filename
        action_filename = data_path

        inputs = []
        actions = []

        with h5py.File(action_filename, "r") as f:
            # Get number of frames
            num_frames = len([k for k in f.keys() if k.startswith('frame_') and k.endswith('_x')])

            for i in range(num_frames):
                # Load RGB
                rgb = f[f'frame_{i}_x'][:]  # (H, W, 3)

                # Load depth if available
                depth_available = f'depth/frame_{i}' in f

                if depth_available:
                    depth = f[f'depth/frame_{i}'][:]  # (H, W)
                    # Add channel dimension: (H, W, 1)
                    depth = np.expand_dims(depth, axis=-1)
                    # Concatenate: (H, W, 4)
                    frame = np.concatenate([rgb, depth], axis=-1)
                else:
                    # Fallback: use RGB only
                    frame = rgb

                inputs.append(frame)

                # Load action
                action = f[f'frame_{i}_y'][:]
                actions.append(action)

        inputs = np.array(inputs, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        print(f"  Loaded {len(inputs)} frames with shape {inputs[0].shape}")

        return inputs, actions

    def get_start_indices_from_data(self, actions_data):
        """Get valid start indices for sequences"""
        seq_len = self.data_config.sequence_length
        return list(range(len(actions_data) - seq_len))

    def get_sequence_from_data(self, inputs, actions, start_index):
        """Extract sequence starting at index"""
        seq_len = self.data_config.sequence_length
        downsample = self.data_config.downsample

        inputs_seq = inputs[start_index:start_index + seq_len:downsample]
        actions_seq = actions[start_index:start_index + seq_len:downsample]

        return inputs_seq, actions_seq
