#!/usr/bin/env python
"""Inspect CS:GO HDF5 data - unpack frames and sequences"""

import h5py
import numpy as np
from pathlib import Path

HDF5_FILE = r"C:\csgo_data\hdf5_dm_july2021_1.hdf5"

print("Inspecting HDF5 data structure...")
print(f"File: {HDF5_FILE}\n")

with h5py.File(HDF5_FILE, 'r') as f:
    # List all keys
    print("Available keys:")
    keys = list(f.keys())
    print(f"  Total frames: {len(keys) // 4}")  # Each frame has 4 keys
    print()

    # Get first frame
    frame_0_x = f['frame_0_x'][:]  # Image
    frame_0_y = f['frame_0_y'][:]  # Actions
    frame_0_xaux = f['frame_0_xaux'][:]  # Aux (health, ammo, team)
    frame_0_helperarr = f['frame_0_helperarr'][:]  # Kill/death flags

    print("Frame 0 structure:")
    print(f"  frame_0_x (image): {frame_0_x.shape} dtype={frame_0_x.dtype}")
    print(f"    Min: {frame_0_x.min()}, Max: {frame_0_x.max()}")
    print(f"  frame_0_y (actions): {frame_0_y.shape} dtype={frame_0_y.dtype}")
    print(f"    Actions: {frame_0_y}")
    print(f"  frame_0_xaux (aux): {frame_0_xaux.shape} dtype={frame_0_xaux.dtype}")
    print(f"  frame_0_helperarr (kill/death): {frame_0_helperarr}")
    print()

    # Extract sequence of 100 frames
    print("Extracting sequence of 100 frames (0-99)...")
    sequence_length = 100
    frames = []
    actions = []

    for i in range(sequence_length):
        frame = f[f'frame_{i}_x'][:]
        action = f[f'frame_{i}_y'][:]
        frames.append(frame)
        actions.append(action)

    frames = np.array(frames)  # Shape: (100, H, W, 3)
    actions = np.array(actions)  # Shape: (100, num_action_dims)

    print(f"Sequence of {sequence_length} frames:")
    print(f"  Frames: {frames.shape} (frames, height, width, channels)")
    print(f"  Actions: {actions.shape} (frames, action_dims)")
    print()

    # Stats
    print("Sequence statistics:")
    print(f"  Frame min/max: {frames.min()}/{frames.max()}")
    print(f"  Action range: {actions.min():.3f} to {actions.max():.3f}")
    print()

    # Sample actions
    print("Sample actions (first 5 frames):")
    for i in range(min(5, len(actions))):
        print(f"  Frame {i}: {actions[i]}")

print("\nData ready for training!")
print(f"Each file has 1000 frames (~60 seconds at 16.7 FPS)")
print(f"Sequence length for training: typically 100 frames (~6 seconds)")
