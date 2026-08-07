#!/usr/bin/env python
"""Check HDF5 files for corruption"""

import h5py
from pathlib import Path
import sys

data_dir = Path(r"C:\csgo_data")
hdf5_files = sorted(data_dir.glob("hdf5_dm_july2021_*.hdf5"))

print(f"Checking {len(hdf5_files)} HDF5 files...\n")

good_files = []
bad_files = []

for hdf5_file in hdf5_files:
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Try to read a frame
            _ = f['frame_0_x'][:]
        print(f"✓ {hdf5_file.name}")
        good_files.append(hdf5_file.name)
    except Exception as e:
        print(f"✗ {hdf5_file.name}: {str(e)[:60]}")
        bad_files.append(hdf5_file.name)

print(f"\n{'='*60}")
print(f"Good files: {len(good_files)}")
print(f"Bad files: {len(bad_files)}")
print(f"{'='*60}")

if bad_files:
    print(f"\nCorrupted files to delete:")
    for f in bad_files:
        print(f"  - {f}")

    # Create cleaned split file
    split_file = Path(r"C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games\pixelbc\data\paper_data_split\csgo_dm_train_files.txt")

    with open(split_file, 'w') as f:
        for fname in good_files:
            f.write(fname + '\n')

    print(f"\nUpdated split file: {split_file}")
    print(f"Now contains {len(good_files)} files")
else:
    print("\n✓ All HDF5 files are valid!")
