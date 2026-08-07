#!/usr/bin/env python
"""Direct download CS:GO dataset from HuggingFace CDN"""

import urllib.request
import os

OUTPUT_DIR = r"C:\csgo_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Direct download URLs from HuggingFace
BASE_URL = "https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/dataset_aim_expert"

print("Downloading CS:GO aim expert dataset...")
print()

for i in range(1, 11):
    filename = f"hdf5_aim_july2021_expert_{i}.hdf5"
    url = f"{BASE_URL}/{filename}"
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(f"[{i}/10] {filename}...", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"OK ({size_mb:.0f}MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

print()
import glob
files = glob.glob(os.path.join(OUTPUT_DIR, "*.hdf5"))
print(f"Total files: {len(files)}")
print(f"Location: {OUTPUT_DIR}")
print()
print("Ready to train!")
