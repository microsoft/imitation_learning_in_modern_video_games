#!/usr/bin/env python
"""List all files in HuggingFace CS:GO repo."""
from huggingface_hub import list_repo_files

repo_id = "TeaPearce/CounterStrike_Deathmatch"
print(f"Listing files in {repo_id}...\n")

try:
    files = list_repo_files(repo_id, repo_type="dataset")

    hdf5_files = [f for f in files if f.endswith('.hdf5')]
    other_files = [f for f in files if not f.endswith('.hdf5')]

    print(f"Total files: {len(files)}")
    print(f"HDF5 files: {len(hdf5_files)}")
    print(f"Other files: {len(other_files)}\n")

    if hdf5_files:
        print("HDF5 Files:")
        print("-" * 60)
        for f in sorted(hdf5_files)[:10]:
            print(f"  {f}")
        if len(hdf5_files) > 10:
            print(f"  ... and {len(hdf5_files) - 10} more")

    if other_files:
        print("\nOther Files:")
        print("-" * 60)
        for f in sorted(other_files):
            print(f"  {f}")

except Exception as e:
    print(f"Error: {e}")
