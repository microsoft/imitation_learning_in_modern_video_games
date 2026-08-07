#!/usr/bin/env python
"""Download CS:GO dataset from HuggingFace and extract tar files"""

from huggingface_hub import hf_hub_download
import tarfile
import os
import glob

OUTPUT_DIR = r'C:\csgo_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download first 3 tar archives (aim expert + first 2 DM batches)
files_to_download = [
    'dataset_aim_expert.tar',
    'hdf5_dm_july2021_1_to_200.tar',
    'hdf5_dm_july2021_201_to_400.tar',
]

print('Downloading and extracting CS:GO dataset...\n')

for i, filename in enumerate(files_to_download, 1):
    print(f'[{i}/{len(files_to_download)}] {filename}...')
    try:
        path = hf_hub_download(
            'TeaPearce/CounterStrike_Deathmatch',
            filename=filename,
            repo_type='dataset',
            local_dir=OUTPUT_DIR
        )
        size_gb = os.path.getsize(path) / (1024**3)
        print(f'  Downloaded: {size_gb:.2f}GB')

        print(f'  Extracting...')
        with tarfile.open(path) as tar:
            tar.extractall(OUTPUT_DIR)
        print(f'  Extracted')
        os.remove(path)

    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

print('\nFiles ready:')
hdf5s = glob.glob(os.path.join(OUTPUT_DIR, '*.hdf5'))
print(f'HDF5 files: {len(hdf5s)}')

if hdf5s:
    total_gb = sum(os.path.getsize(f) for f in hdf5s) / (1024**3)
    print(f'Total size: {total_gb:.1f}GB')
    print(f'\nReady to train!')
else:
    print('No HDF5 files extracted')
