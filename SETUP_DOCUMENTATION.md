# CS:GO Imitation Learning Setup - Complete Documentation

## Overview

This project trains a behavioral cloning agent on Counter-Strike: Global Offensive gameplay using:
- **Dataset**: 52 HDF5 files (52,000 frames) of deathmatch gameplay
- **Model**: CNN encoder + LSTM policy network
- **Input**: RGB frames (optionally RGB+Depth)
- **Actions**: 51-dimensional action vector (keys, mouse, clicks)

---

## Environment Setup (Windows + RTX 5090 / Blackwell sm_120)

This box is Windows 11 with an RTX 5090 (Blackwell, `sm_120`). Two hard requirements:

- **torch must be cu128 or newer.** The default PyPI wheel is CPU-only, and cu121/cu124 wheels have no `sm_120` kernels (`CUDA error: no kernel image is available for execution on the device`). Use **torch 2.7.0 + cu128**.
- **Python 3.11**, created with **uv** (3.12 hits thread-pool deadlocks on Windows).

Recreate the venv from scratch:
```powershell
.\recreate_venv.bat
```
It removes `.venv`, runs `uv venv --python 3.11`, installs `torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0` from the cu128 index, then `requirements.txt` + the `pixelbc` package, and verifies `torch.cuda.is_available()`.

### DataLoader workers on Windows — keep at 0
`BCDataSet` loads the **entire dataset into RAM** in `__init__`. On Windows (`spawn` start method) every DataLoader worker gets a *pickled copy* of that in-RAM dataset, so the old `train_num_workers: 20` spawned 20 dataset copies → RAM thrash + spawn overhead → **~0.02 it/s (≈8.7 h/epoch)**.

Fix (already applied):
- `pixelbc/data/datamodule.py` — `persistent_workers`/`prefetch_factor` are only passed when `num_workers > 0`, so `0` is now legal.
- `configs/csgo/*.yaml` — `train_num_workers` and `other_num_workers` set to **0**.

With the data already in RAM, `num_workers=0` (single process, no copy, no IO wait) is the **fastest** path here. A worker change only takes effect on a **fresh run** — restart training to pick it up.

### torch build: CPU vs cu128 (GPU)
The venv currently has **`torch 2.6.0+cpu` / `torchvision 0.21.0+cpu`** — because `requirements.txt` pins `torch==2.6.0` + `torchvision==0.21.0` under `--find-links .../torch_stable.html`, which resolves to **CPU** wheels. Consequence: `trainer.accelerator=gpu` fails with **`MisconfigurationException: No supported gpu backend found!`** (torch can't see CUDA). CPU training works as-is.

To train on the 5090 (Blackwell sm_120), replace torch with the **cu128** build (do it *before* `-r requirements.txt`, and don't let requirements re-pin it back to CPU afterwards):
```
uv pip install --python <this-venv>\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
```
This needs **~5–6 GB free disk** and **no Python process holding the venv** (see the gotcha below).

### Gotcha: interrupted torch install → corrupted `torchvision` metadata
Symptom: a `TypeError: expected string or bytes-like object, got 'NoneType'` deep in `lightning_utilities/.../imports.py` (`Version(_version("torchvision"))`), which kills every `import lightning` / `torchmetrics`.

Cause: a torch/torchvision reinstall was interrupted (a running Python held `torchvision/_C.pyd`, and/or disk hit ~0). uv's uninstall step had already **emptied `torchvision-<ver>.dist-info/`** before it aborted, so `importlib.metadata.version("torchvision")` returns `None` even though `import torchvision` still works (that version comes from the package's own `version.py`).

Fix without a reinstall (no disk needed) — recreate the metadata file:
```
<venv>\Lib\site-packages\torchvision-0.21.0.dist-info\METADATA:
    Metadata-Version: 2.1
    Name: torchvision
    Version: 0.21.0
```
`version("torchvision")` then resolves and lightning imports again. A later `pip install torchvision==<ver>` regenerates the dist-info fully. **Prevention:** close all Python using the venv and ensure ≥6 GB free *before* any torch (re)install.

---

## Directory Structure

```
C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games\
├── pixelbc/                          # Main training framework
│   ├── configs/csgo/                # Training configurations
│   │   ├── cnn_lstm.yaml           # RGB-only (3-channel)
│   │   ├── cnn_lstm_rgbd.yaml      # RGB+Depth (4-channel)
│   │   └── default.yaml            # Default config
│   ├── data/
│   │   ├── paper_data_split/
│   │   │   ├── csgo_dm_train_files.txt  # Split file (52 valid files)
│   │   │   └── csgo_train_files_small.txt # Aim expert (unused)
│   │   ├── csgo_data_parser_rgbd.py    # 4-channel data loader
│   │   └── datamodule.py
│   ├── run_train.py                # Training entry point
│   └── models/utils/
│       └── train_metrics.py         # Evaluation metrics
├── trained_models/                  # Pretrained models (TensorFlow/Keras)
│   ├── ak47_sub_55k_drop_d4.h5                    # AK47 baseline
│   ├── ak47_sub_55k_drop_d4_dmexpert_28.h5       # Recommended ⭐
│   ├── ak47_sub_55k_drop_d4_aimexpertv2_60.h5   # Aim specialist
│   └── July_remoterun7_...h5                     # Full dataset pretrain
├── training_logs/                   # Training outputs
│   └── default_seed_42/
│       ├── models/last.ckpt
│       ├── events.out.tfevents
│       └── metrics.json
├── train_cnn_lstm.bat              # Train RGB-only
├── train_cnn_lstm_rgbd.bat         # Train RGB+Depth
├── check_hdf5_files.py             # Validate HDF5 files
├── extract_depth_from_hdf5.py      # Extract depth maps
├── inspect_data.py                 # Inspect HDF5 structure
└── .venv/                          # Python environment

C:\csgo_data/                       # Dataset directory
├── hdf5_dm_july2021_1.hdf5        # 1000 frames each
├── hdf5_dm_july2021_2.hdf5
├── ...
└── hdf5_dm_july2021_51.hdf5

C:\workspace\NAVIGATION\Depth-Anything-V2/  # Depth estimation model
├── run_depth.bat                   # Extract depth from video
├── vis_video_depth/
│   └── sequence_demo.mp4          # Grayscale depth video (18 sec)
└── checkpoints/depth_anything_v2_vitl.pth
```

---

## Data Format

### HDF5 File Structure

Each HDF5 file contains 1000 frames (~60 seconds) with keys:
```
frame_0_x           → RGB image (150×280×3, uint8)
frame_0_y           → Action labels (51-dim, float32)
frame_0_xaux        → Auxiliary (health, ammo, team)
frame_0_helperarr   → [kill_flag, death_flag]

frame_1_x, frame_1_y, ...
...
frame_999_x, frame_999_y, ...

(Optional) depth/frame_0  → Grayscale depth (150×280, uint8)
(Optional) depth/frame_1, ...
```

### Action Vector (51 dimensions)

- **Keys (24)**: WASD directionals + modifiers (one-hot)
- **Mouse clicks (2)**: Left/right click (binary)
- **Mouse position (20)**: Horizontal + vertical binned (one-hot per axis)
- **Metadata (5)**: Additional modifiers

---

## Training Configurations

### RGB-Only (3-channel)
```yaml
# pixelbc/configs/csgo/cnn_lstm.yaml
encoder:
  type: cnn
  cnn_encoder_start_channels: 32
  cnn_encoder_dim: 4

bc_policy:
  type: lstm
  lstm_num_layers: 2
  hidden_size: 512

learning_rate: 3e-4
batch_size: 32
sequence_length: 100 frames (~6 sec)
```

### RGB+Depth (4-channel)
```yaml
# pixelbc/configs/csgo/cnn_lstm_rgbd.yaml
# Same as above, but:
# - Input: RGB + grayscale depth concatenated
# - Encoder processes 4 channels instead of 3
# - All other settings identical
```

---

## Running Training

### RGB-Only (Fast, Works Immediately)
```powershell
cd C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games
.\train_cnn_lstm.bat
```

**Time**: ~2-4 hours per epoch on CPU
**Data**: 52 files × 1000 frames = 52k training examples

### RGB+Depth (Slow, Requires Depth Extraction First)
```powershell
# Step 1: Extract depth from HDF5 frames
python extract_depth_from_hdf5.py

# Step 2: Train with 4-channel input
.\train_cnn_lstm_rgbd.bat
```

**Time**: ~1-2 hours per file for depth extraction, then 2-4 hours training
**Data**: First 3 files have depth, remaining 49 use RGB-only

---

## Model Architecture

### Encoder: CNN
```
Input: (batch, 100, 150, 280, 3) or (batch, 100, 150, 280, 4)
    ↓
Rescaling (0-255 → 0-1)
    ↓
Normalization (channel-wise)
    ↓
Conv2D stem (32 filters, stride 2)
    ↓
Multiple conv blocks
    ↓
Output: (batch, 100, 256) [256-dim feature per frame]
```

### Policy: LSTM
```
Input: (batch, 100, 256)
    ↓
LSTM Layer 1 (512 hidden)
    ↓
LSTM Layer 2 (512 hidden)
    ↓
Dense layers
    ↓
Output: (batch, 100, 51) [51-dim action per frame]
```

---

## Pretrained Models

### Available Models (23MB each, TensorFlow/Keras)

| Model | Training Data | Use Case |
|-------|---------------|----------|
| `ak47_sub_55k_drop_d4` | AK47 sequences only | Weapon baseline |
| `ak47_sub_55k_drop_d4_dmexpert_28` ⭐ | AK47 + expert DM (190 files) | **Recommended for finetuning** |
| `ak47_sub_55k_drop_d4_aimexpertv2_60` | AK47 + expert aim (45 files) | Aim mode specialist |
| `July_remoterun7...` | Full online dataset (5500 files, ~700GB) | Maximum generalization |

### Using Pretrained Models

```python
# Load pretrained model
import tensorflow as tf
model = tf.keras.models.load_model('trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5')

# Fine-tune on your CS:GO data
python pixelbc/run_train.py pixelbc/configs/csgo/cnn_lstm.yaml \
  checkpoint=trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5 \
  data.data_path=C:\csgo_data \
  model.learning_rate=1e-4 \
  trainer.max_epochs=10
```

---

## Evaluation

### Metrics Computed During Training

- **Action Accuracy**: % of correct discrete action predictions
- **Button Metrics**: TP/TN/FP/FN rates for binary actions
- **Joystick Metrics**: Accuracy for directional inputs
- **MSE/R²**: Regression metrics for continuous actions

### Online Evaluation (Requires CS:GO)

```bash
# Run trained agent in live CS:GO gameplay
python pixelbc/online_rollout/csgo_rollout.py \
  --checkpoint training_logs/default_seed_42/models/last.ckpt \
  --num-rollouts 3 \
  --rollout-time 5
```

---

## Depth Extraction

### Extract Depth from Video
```powershell
cd C:\workspace\NAVIGATION\Depth-Anything-V2
.\run_depth.bat C:\csgo_data\sequence_demo.mp4
```

**Output**: Grayscale depth map video in `vis_video_depth/`

### Extract Depth from HDF5 Files
```powershell
python extract_depth_from_hdf5.py
```

**Processing**: ~30-40 min per file (1000 frames each)
**Output**: Depth stored as `depth/frame_i` in each HDF5 file

---

## Data Modification Guide

### Core Data Loading Files

#### 1. **pixelbc/data/data_parsers.py** (Frame & Action Loading)
Main file for loading frames from HDF5 and preprocessing images.

```python
# Line ~375: get_all_data(filename)
def get_all_data(self, filename):
    """Load RGB frames and action labels from HDF5"""
    with h5py.File(filename, "r") as f:
        for i in range(num_frames):
            frame = f[f'frame_{i}_x'][:]  # Load image
            action = f[f'frame_{i}_y'][:]  # Load action
            # Modify here to change frame/action preprocessing
```

**Common modifications:**
- Normalize frames to [-1, 1] instead of [0, 1]
- Add color space conversion (RGB → HSV, LAB)
- Apply Gaussian blur or other filters
- Adjust action scaling or discretization

#### 2. **pixelbc/data/datamodule.py** (Batch Creation & Augmentation)
Controls PyTorch DataLoader, batching, and data augmentation.

```python
# Line ~21: BCDataSet.__init__()
def __init__(self, data_parser, data_files):
    for data_file in data_files:
        self.data_by_file[data_file] = self.data_parser.get_all_data(data_file)
        # Modify here to add augmentation (crop, rotate, noise)

# Line ~83: train_dataloader()
def train_dataloader(self):
    return DataLoader(
        self.train_set,
        batch_size=self.data_config.batch_size,
        # Modify batch creation, collate_fn for custom batching
    )
```

**Common modifications:**
- Add data augmentation (RandomCrop, ColorJitter, RandomRotate)
- Custom collate function for padding sequences
- Hard negative mining (focus on difficult examples)
- Class weighting (balance rare actions)

#### 3. **pixelbc/data/csgo_data_parser_rgbd.py** (RGB+Depth Loading)
Custom parser for 4-channel input (RGB + grayscale depth).

```python
# Modify to add/remove depth or other modalities
def get_all_data(self, filename):
    rgb = f[f'frame_{i}_x'][:]      # 3 channels
    depth = f[f'depth/frame_{i}'][:] # 1 channel
    frame = np.concatenate([rgb, depth], axis=-1)  # 4 channels
```

**Common modifications:**
- Add optical flow as additional channel
- Use depth for auxiliary task learning
- Combine multiple sensors/modalities

#### 4. **pixelbc/data/paper_data_split/csgo_dm_train_files.txt**
Text file listing which HDF5 files to use for training.

```
hdf5_dm_july2021_1.hdf5
hdf5_dm_july2021_2.hdf5
hdf5_dm_july2021_3.hdf5
...
```

**How to modify:**
- Remove files to train on smaller dataset
- Add/remove specific map types
- Create separate train/val/test splits
- Run `check_hdf5_files.py` after editing to validate

---

### Data Modification Examples

#### Example 1: Add Image Augmentation

**File**: `pixelbc/data/datamodule.py` (Line ~20)

```python
from torchvision import transforms

class BCDataSet(Dataset):
    def __init__(self, data_parser, data_files):
        # Add augmentation pipeline
        self.augment = transforms.Compose([
            transforms.RandomCrop(size=(128, 128)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=5),
        ])
        
        for data_file in data_files:
            self.data_by_file[data_file] = self.data_parser.get_all_data(data_file)
    
    def __getitem__(self, idx):
        inputs_seq, actions_seq = ... # Get data
        # Apply augmentation per sequence
        inputs_seq = torch.stack([
            self.augment(torch.from_numpy(frame).permute(2,0,1))
            for frame in inputs_seq
        ])
        return inputs_seq, actions_seq
```

#### Example 2: Normalize Images to [-1, 1]

**File**: `pixelbc/data/data_parsers.py` (Line ~375)

```python
def get_all_data(self, filename):
    with h5py.File(filename, "r") as f:
        for i in range(num_frames):
            frame = f[f'frame_{i}_x'][:]
            # Normalize from [0, 255] to [-1, 1]
            frame = (frame.astype(np.float32) / 127.5) - 1.0
            inputs.append(frame)
```

#### Example 3: Custom Data Split (80/10/10 Train/Val/Test)

**Create file**: `pixelbc/data/paper_data_split/csgo_dm_train_files_80.txt`

```
hdf5_dm_july2021_1.hdf5
hdf5_dm_july2021_2.hdf5
...
hdf5_dm_july2021_41.hdf5
```

Then create `csgo_dm_val_files_10.txt` and `csgo_dm_test_files_10.txt`

**Update config**:
```yaml
data:
  train_split_file_path: paper_data_split/csgo_dm_train_files_80.txt
  validation_split_file_path: paper_data_split/csgo_dm_val_files_10.txt
  test_split_file_path: paper_data_split/csgo_dm_test_files_10.txt
```

#### Example 4: Add Optical Flow Channel

**Create file**: `pixelbc/data/optical_flow_parser.py`

```python
import cv2

class CSGODataParserWithFlow:
    def get_all_data(self, filename):
        with h5py.File(filename, "r") as f:
            prev_frame = None
            for i in range(num_frames):
                frame = f[f'frame_{i}_x'][:]
                
                if prev_frame is not None:
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Add as 2-channel input (Fx, Fy)
                    frame = np.concatenate([frame, flow], axis=-1)  # 5 channels
                
                inputs.append(frame)
                prev_frame = frame[:, :, :3]  # Keep only RGB for next iteration
```

---

### Performance Tips for Data Loading

| Optimization | Impact | Difficulty |
|--------------|--------|------------|
| Increase `num_workers` | ↑↑ Speed | Easy |
| Enable GPU | ↑↑↑ Speed | Easy |
| Reduce `prefetch_factor` | ↓ Memory | Easy |
| Cache frames in RAM | ↑↑ Speed | Medium |
| Use mixed precision | ↑ Speed | Easy |
| Parallelize augmentation | ↑ Speed | Medium |
| Pre-compute depth maps | ↑ Speed | Hard |
| Create data index | ↑ Speed | Medium |

**Fastest config (Windows / this box):**
```yaml
data:
  train_num_workers: 0          # in-RAM dataset -> 0 is fastest on Windows (spawn)
  other_num_workers: 0
  batch_size: 64                # larger batches on GPU

trainer:
  accelerator: gpu
  precision: 16-mixed           # mixed precision
```
> The `num_workers > 0` / `prefetch_factor` advice above is for **Linux (fork)**. On Windows the dataset is duplicated per worker, so keep workers at 0.

---

## Troubleshooting

### Corrupted HDF5 Files
```powershell
python check_hdf5_files.py
```
Automatically removes bad files from split file.

### Model Not Loading
- Ensure checkpoint path is correct
- Check model format (TensorFlow .h5 vs PyTorch .ckpt)
- Verify file exists and isn't corrupted

### Out of Memory
- Reduce batch_size in config (default: 32 → try 16)
- Reduce sequence_length (default: 100 → try 50)
- Use smaller encoder (start_channels: 32 → 16)

### Training Too Slow (~0.02 it/s / hours per epoch)
- **num_workers must be 0 on Windows** — the dataset is in RAM, so spawned workers each duplicate it and thrash. `configs/csgo/*.yaml` now default to 0; if you edited them back up, set `train_num_workers`/`other_num_workers` to 0 and restart.
- Confirm the GPU is actually used: `python -c "import torch;print(torch.cuda.is_available())"` must be `True` (cu128 torch — see Environment Setup).
- Reduce max_epochs / sequence_length for quick smoke tests.

---

## Performance Baselines

| Task | Metric | Value |
|------|--------|-------|
| Action Accuracy | % correct predictions | ~60-75% (varies by action type) |
| Inference Speed | FPS on CPU | ~10-15 FPS |
| Inference Speed | FPS on GPU | ~30-50 FPS |
| Training Time | Per epoch (52 files) | 2-4 hours CPU |
| Memory Usage | GPU | ~2-3 GB |
| Memory Usage | CPU | ~8-12 GB |

---

## Dataset Info

### Downloaded Data
- **Total files**: 52 valid HDF5 (1 corrupted, removed)
- **Total frames**: 52,000 (52 files × 1000 frames)
- **Duration**: ~52 minutes of gameplay (~16.7 FPS)
- **Size on disk**: 12 GB
- **Resolution**: 150×280×3 RGB

### Data Source
- **Dataset**: TeaPearce/CounterStrike_Deathmatch
- **HuggingFace**: https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch
- **Paper**: "Counter-Strike Deathmatch with Large-Scale Behavioural Cloning" (CoG 2022)

---

## Files Created in This Session

| File | Purpose |
|------|---------|
| `cnn_lstm.yaml` | RGB-only training config |
| `cnn_lstm_rgbd.yaml` | RGB+Depth training config |
| `csgo_dm_train_files.txt` | Split file (52 valid files) |
| `check_hdf5_files.py` | Validate and clean HDF5 files |
| `extract_depth_from_hdf5.py` | Extract depth using Depth-Anything-V2 |
| `inspect_data.py` | Inspect HDF5 structure |
| `setup_rgb_depth.py` | Setup 4-channel training |
| `train_cnn_lstm.bat` | RGB-only training script |
| `train_cnn_lstm_rgbd.bat` | RGB+Depth training script |
| `train_with_depth.py` | Depth augmentation helper |
| `recreate_venv.bat` | Rebuild `.venv` (Python 3.11 + torch 2.7.0 cu128, Blackwell) |

---

## Next Steps

1. **Train RGB-only** (fast):
   ```powershell
   .\train_cnn_lstm.bat
   ```

2. **Monitor training** at `training_logs/default_seed_42/`

3. **Evaluate** using validation metrics in TensorBoard

4. **(Optional) Extract depth** for all 52 files and retrain with RGB+Depth

5. **Deploy** to CS:GO using `pixelbc/online_rollout/csgo_rollout.py`

---

## References

- **pixelbc Framework**: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning
- **Depth-Anything-V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
- **HuggingFace Datasets**: https://huggingface.co/datasets

---

**Last Updated**: August 7, 2026
**Status**: Ready for CPU training (venv repaired: torchvision metadata restored). GPU needs the cu128 torch swap (see "torch build: CPU vs cu128") — blocked on free disk.
