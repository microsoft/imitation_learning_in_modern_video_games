#!/usr/bin/env python
"""Download CS:GO dataset using huggingface_hub with advanced approaches"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

REPO_ID = "TeaPearce/CounterStrike_Deathmatch"
OUTPUT_DIR = r"C:\csgo_data"

print("CS:GO Dataset Download via huggingface_hub")
print("=" * 60)
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("Method 1: snapshot_download with force_download...")
    print()

    # Try snapshot_download which handles LFS if git-lfs is available
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
        allow_patterns=["*.hdf5"],
    )

    print(f"Downloaded to: {local_dir}")
    print()

    # Check what was downloaded
    hdf5_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".hdf5")]

    if hdf5_files:
        print(f"✓ Downloaded {len(hdf5_files)} HDF5 files")
        total_size_gb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in hdf5_files) / (1024**3)
        print(f"  Total size: {total_size_gb:.1f}GB")
        print()
        print("Ready to train!")
    else:
        print("✗ No HDF5 files found after download")
        print()
        print("Downloaded files:")
        for f in os.listdir(OUTPUT_DIR)[:30]:
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / 1024
                print(f"  - {f} ({size:.1f}KB)")

except Exception as e:
    print(f"Error: {e}")
    print()
    print("Trying Method 2: hf_hub_download for individual files...")
    print()

    try:
        # List all files in dataset
        files = list_repo_files(repo_id=REPO_ID, repo_type="dataset", recursive=True)
        hdf5_files = [f for f in files if f.endswith(".hdf5")]

        print(f"Found {len(hdf5_files)} HDF5 files in repo")

        if hdf5_files:
            # Download first 10
            for i, file_path in enumerate(hdf5_files[:10], 1):
                print(f"[{i}] {Path(file_path).name}...", end=" ", flush=True)

                try:
                    local_path = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=OUTPUT_DIR,
                        local_dir_use_symlinks=False,
                        force_download=True,
                    )
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    print(f"OK ({size_mb:.1f}MB)")

                except Exception as e2:
                    print(f"FAILED: {str(e2)[:50]}")

            print()
            downloaded = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".hdf5")]
            if downloaded:
                print(f"Downloaded {len(downloaded)} files")
            else:
                print("ERROR: No files downloaded")
        else:
            print("ERROR: No HDF5 files found in repository")

    except Exception as e2:
        print(f"Method 2 also failed: {e2}")
        sys.exit(1)

except KeyboardInterrupt:
    print("\nInterrupted by user")
    sys.exit(1)
