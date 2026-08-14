#!/usr/bin/env python
"""Download and extract all CS:GO datasets from HuggingFace."""
import os
import tarfile
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def extract_tar(tar_path, extract_to):
    """Extract tar file."""
    print(f"  Extracting: {tar_path.name}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
    print(f"  ✓ Extracted")

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"  Extracting: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(path=extract_to)
    print(f"  ✓ Extracted")

def main():
    base_dir = Path("C:\\csgo_data")
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / "temp_archives"
    temp_dir.mkdir(exist_ok=True)

    repo_id = "TeaPearce/CounterStrike_Deathmatch"

    print("CS:GO Dataset Downloader")
    print(f"Base directory: {base_dir}\n")

    # Files to download - ALL 37 archives
    archives = [
        "dataset_aim_expert.tar",
        "hdf5_dm_july2021_1_to_200.tar",
        "hdf5_dm_july2021_201_to_400.tar",
        "hdf5_dm_july2021_401_to_600.tar",
        "hdf5_dm_july2021_601_to_800.tar",
        "hdf5_dm_july2021_801_to_1000.tar",
        "hdf5_dm_july2021_1001_to_1200.tar",
        "hdf5_dm_july2021_1201_to_1400.tar",
        "hdf5_dm_july2021_1401_to_1600.tar",
        "hdf5_dm_july2021_1601_to_1800.tar",
        "hdf5_dm_july2021_1801_to_2000.tar",
        "hdf5_dm_july2021_2001_to_2200.tar",
        "hdf5_dm_july2021_2201_to_2400.tar",
        "hdf5_dm_july2021_2401_to_2600.tar",
        "hdf5_dm_july2021_2601_to_2800.tar",
        "hdf5_dm_july2021_2801_to_3000.tar",
        "hdf5_dm_july2021_3001_to_3200.tar",
        "hdf5_dm_july2021_3201_to_3400.tar",
        "hdf5_dm_july2021_3401_to_3600.tar",
        "hdf5_dm_july2021_3601_to_3800.tar",
        "hdf5_dm_july2021_3801_to_4000.tar",
        "hdf5_dm_july2021_4001_to_4200.tar",
        "hdf5_dm_july2021_4201_to_4400.tar",
        "hdf5_dm_july2021_4401_to_4600.tar",
        "hdf5_dm_july2021_4601_to_4800.tar",
        "hdf5_dm_july2021_4801_to_5000.tar",
        "hdf5_dm_july2021_5001_to_5200.tar",
        "hdf5_dm_july2021_5201_to_5400.tar",
        "hdf5_dm_july2021_5401_to_5500.tar",
    ]

    print(f"Downloading {len(archives)} archives...\n")

    for archive_name in archives:
        try:
            print(f"Downloading: {archive_name}")
            archive_path = hf_hub_download(
                repo_id=repo_id,
                filename=archive_name,
                repo_type="dataset",
                local_dir=str(temp_dir),
            )
            print(f"  ✓ Downloaded to {archive_path}")

            # Extract
            extract_tar(Path(archive_path), base_dir)

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue

    # Cleanup temp
    print(f"\nCleaning up temp directory...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    # List all files
    print(f"\n{'='*60}")
    print("Downloaded HDF5 files:")
    print(f"{'='*60}")
    hdf5_files = sorted(base_dir.glob("*.hdf5"))
    print(f"Total files: {len(hdf5_files)}\n")

    if len(hdf5_files) <= 20:
        for f in hdf5_files:
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.name:50} {size_gb:6.2f} GB")
    else:
        for f in list(hdf5_files)[:10]:
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.name:50} {size_gb:6.2f} GB")
        print(f"  ... and {len(hdf5_files) - 10} more")

    total_size = sum(f.stat().st_size for f in hdf5_files) / (1024**3)
    print(f"\nTotal size: {total_size:.2f} GB")

    return 0

if __name__ == "__main__":
    exit(main())
