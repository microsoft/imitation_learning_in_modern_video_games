#!/usr/bin/env python
"""Extract frames from HDF5, run Depth-Anything-V2, add depth frames back"""

import h5py
import numpy as np
from pathlib import Path
import sys
import os

# Add Depth-Anything-V2 to path
sys.path.insert(0, r"C:\workspace\NAVIGATION\Depth-Anything-V2")

try:
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"Error importing Depth-Anything-V2: {e}")
    sys.exit(1)

def load_depth_model(encoder='vitl'):
    """Load pretrained depth model"""
    print(f"Loading Depth-Anything-V2 model ({encoder})...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model_path = r"C:\workspace\NAVIGATION\Depth-Anything-V2\checkpoints\depth_anything_v2_vitl.pth"

    if not Path(model_path).exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device).eval()

    print("Model loaded successfully\n")
    return model, device


def extract_and_add_depth(hdf5_file, model, device, num_frames=1000):
    """Extract frames from HDF5, run depth estimation, add back to HDF5"""

    print(f"Processing: {hdf5_file}")

    frames = []
    depths = []

    # Extract frames
    print(f"  Extracting {num_frames} frames from HDF5...")
    with h5py.File(hdf5_file, 'r') as f:
        for i in range(num_frames):
            try:
                frame = f[f'frame_{i}_x'][:]
                frames.append(frame)
            except KeyError:
                break

    print(f"  Extracted {len(frames)} frames")

    # Run depth estimation
    print(f"  Running depth estimation...")
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(frames)}")

            # Infer depth
            depth = model.infer_image(frame, input_size=518)

            # Normalize to 0-255
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
            depths.append(depth_norm)

    # Add depth frames to HDF5
    print(f"  Adding {len(depths)} depth frames to HDF5...")
    with h5py.File(hdf5_file, 'r+') as f:
        # Create depth group if not exists
        if 'depth' in f:
            del f['depth']

        depth_group = f.create_group('depth')

        for i, depth in enumerate(depths):
            depth_group.create_dataset(
                f'frame_{i}',
                data=depth,
                compression='gzip',
                compression_opts=4
            )

    print(f"  ✓ Complete: {len(depths)} depth frames added\n")


def main():
    data_dir = Path(r"C:\csgo_data")
    hdf5_files = sorted(data_dir.glob("hdf5_dm_july2021_*.hdf5"))

    if not hdf5_files:
        print("ERROR: No HDF5 files found in C:\\csgo_data")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} HDF5 files\n")

    # Load model once
    model, device = load_depth_model('vitl')

    # Process first N files (change as needed)
    num_to_process = 3  # Start with 3 files

    for i, hdf5_file in enumerate(hdf5_files[:num_to_process], 1):
        print(f"[{i}/{num_to_process}]")
        try:
            extract_and_add_depth(str(hdf5_file), model, device, num_frames=1000)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    print("="*60)
    print(f"Processed {num_to_process} files")
    print("Depth frames added to HDF5 as 'depth/frame_i' datasets")
    print("="*60)


if __name__ == '__main__':
    main()
