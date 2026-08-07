#!/usr/bin/env python
"""Extract frames from HDF5 and save as video"""

import h5py
import cv2
import numpy as np
from pathlib import Path

HDF5_FILE = r"C:\csgo_data\hdf5_dm_july2021_1.hdf5"
OUTPUT_VIDEO = r"C:\csgo_data\sequence_demo.mp4"

print(f"Extracting frames from: {HDF5_FILE}")
print(f"Output video: {OUTPUT_VIDEO}\n")

# Extract frames
frames = []
with h5py.File(HDF5_FILE, 'r') as f:
    # Get first 300 frames (18 seconds at 16.7 FPS)
    num_frames = 300
    print(f"Extracting {num_frames} frames...")

    for i in range(num_frames):
        frame = f[f'frame_{i}_x'][:]
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

print(f"Loaded {len(frames)} frames")
print(f"Frame shape: {frames[0].shape}")

# Save as video
height, width = frames[0].shape[:2]
fps = 16.67  # Frames per second from dataset

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"\nWriting video ({width}x{height} @ {fps} FPS)...")
for i, frame in enumerate(frames):
    out.write(frame)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(frames)} frames written")

out.release()
print(f"\n✓ Video saved: {OUTPUT_VIDEO}")
print(f"  Duration: {len(frames) / fps:.1f} seconds")
print(f"  Frames: {len(frames)}")
