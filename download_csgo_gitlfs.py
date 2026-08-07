#!/usr/bin/env python
"""Download CS:GO dataset using Git LFS"""

import subprocess
import os
import glob
import shutil
import sys

REPO_URL = "https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch"
TEMP_DIR = r"C:\csgo_repo_tmp"
OUTPUT_DIR = r"C:\csgo_data"
MAX_FILES = 10

print("CS:GO Dataset Download via Git LFS")
print("=" * 60)
print()

# Clean up temp dir if it exists
if os.path.exists(TEMP_DIR):
    print(f"Cleaning up {TEMP_DIR}...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# Clone repo
print(f"Cloning repository...")
try:
    subprocess.run(
        ["git", "clone", REPO_URL, TEMP_DIR],
        check=True,
        timeout=300,
        capture_output=True
    )
except subprocess.CalledProcessError as e:
    print(f"Clone failed: {e}")
    print(e.stderr.decode() if e.stderr else "")
    sys.exit(1)

print("Cloned successfully")
print()

# Configure sparse checkout to download only what we need
print("Configuring sparse checkout...")
os.chdir(TEMP_DIR)

try:
    # Initialize sparse checkout
    subprocess.run(
        ["git", "config", "core.sparseCheckoutCone", "true"],
        check=True,
        capture_output=True
    )

    # Set pattern to only get HDF5 files
    sparse_file = os.path.join(TEMP_DIR, ".git", "info", "sparse-checkout")
    os.makedirs(os.path.dirname(sparse_file), exist_ok=True)
    with open(sparse_file, "w") as f:
        f.write("dataset_aim_expert/*.hdf5\n")

except Exception as e:
    print(f"Sparse checkout setup failed: {e}")

# Pull LFS files
print("Pulling LFS files (this may take a while)...")
print()

try:
    result = subprocess.run(
        ["git", "lfs", "pull", "--include=dataset_aim_expert/*.hdf5"],
        timeout=3600,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ LFS pull successful")
    else:
        print(f"LFS pull warning: {result.stderr[:200]}")

except subprocess.TimeoutExpired:
    print("LFS pull timeout")
except subprocess.CalledProcessError as e:
    print(f"LFS pull failed: {e}")

print()

# Move files to output directory
print(f"Moving files to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

source_dir = os.path.join(TEMP_DIR, "dataset_aim_expert")
if os.path.exists(source_dir):
    hdf5_files = glob.glob(os.path.join(source_dir, "*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")

    for i, src_file in enumerate(hdf5_files[:MAX_FILES], 1):
        filename = os.path.basename(src_file)
        dst_file = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i}/{min(len(hdf5_files), MAX_FILES)}] {filename}...", end=" ", flush=True)

        try:
            shutil.copy(src_file, dst_file)
            size_gb = os.path.getsize(dst_file) / (1024**3)
            print(f"OK ({size_gb:.2f}GB)")
        except Exception as e:
            print(f"FAILED: {e}")
else:
    print(f"ERROR: {source_dir} not found")

# Cleanup
print()
print("Cleaning up temporary directory...")
os.chdir("..")
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print()
print("=" * 60)

# Summary
downloaded = glob.glob(os.path.join(OUTPUT_DIR, "*.hdf5"))
if downloaded:
    total_gb = sum(os.path.getsize(f) for f in downloaded) / (1024**3)
    print(f"✓ Downloaded {len(downloaded)} files ({total_gb:.1f}GB)")
    print()
    print("Ready to train!")
    print(f"  run: train.bat")
else:
    print("✗ No files downloaded")
    sys.exit(1)
