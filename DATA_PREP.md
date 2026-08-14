# CS:GO Data Prep — Full Pipeline

End-to-end steps to go from raw HF tars → a training-ready `C:\csgo_data` with
**12288-d spatial** DINOv2 embeddings, plus every gotcha hit along the way.

Run all commands from the repo root:
`C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games-rgbd`
(uses its own `.venv`). The `_alex` checkout shares this venv.

---

## 0. Current state (as of this doc)

| item | value |
|---|---|
| Data dir | `C:\csgo_data` (single dir — training reads ONE `data.data_path`) |
| Files | **438** `.hdf5` (329 original + 109 new from chunk 401–600) |
| RGB embeddings | `embeddings\dino\dinov2_vitb14\` — **12288-d spatial** (regenerated) |
| Depth embeddings | `embeddings\dino_depth\dinov2_vitb14\` — regenerate at 12288 for RGB+D |
| Train split | `pixelbc\data\paper_data_split\csgo_dm_train_files.txt` — **408** files |
| Val split | `csgo_dm_val_files.txt` — **10** (fixed yardstick, never add to it) |
| Test split | `csgo_dm_test_files.txt` — **20** (fixed) |

---

## 1. Download data (Xet-free streaming)

The HF Xet backend **wedges** when two downloads share its chunk cache. Use the
streaming HTTP downloader instead — it unpacks each `.hdf5` as it arrives and
verifies size (re-downloads truncated files).

```
:: list available chunks (28 total, 200 files each, ~26 GB)
download_csgo_new.bat list

:: stream a fully-NEW gap chunk into a SEPARATE dir (default 500-file cap across chunks)
download_csgo_new.bat hdf5_dm_july2021_401_to_600.tar
```
- Zero-overlap gap chunks for the current set: **401–600, 601–800, 801–1000**, and everything 1201+.
- Lands in `C:\csgo_data_new`. Live `[N/200] file.hdf5 (130 MB) files/s` counter.
- Underlying script: `download_csgo_stream.py` (`--tars`, `--dest`, `--max-files`).

**Disk:** each chunk ~26 GB. `C:` was tight (~28–50 GB free at times). `C:\csgo_dl`
is safe-to-clear Xet staging junk. The **114 GB `LTX-2.5-Diffusers`** HF cache is
the big consumer if you need to reclaim space.

---

## 2. Merge into one directory

Training reads a single `data.data_path`, so merge new files into `C:\csgo_data`.
**Order matters** (delete old embeddings BEFORE moving new ones in):

```
:: 1. move new hdf5 into the main dir
move C:\csgo_data_new\*.hdf5 C:\csgo_data\

:: 2. delete the OLD 768-d embeddings FIRST (they are regenerable)
rmdir /s /q C:\csgo_data\embeddings\dino\dinov2_vitb14

:: 3. move the new 12288 embeddings in
move C:\csgo_data_new\embeddings\dino\dinov2_vitb14 C:\csgo_data\embeddings\dino\
```

---

## 3. Depth maps (`make_depth.py`) — RGB+D only

Injects `frame_i_depth` into each hdf5 via **Depth-Anything-V2 `vits`** (do NOT
change the encoder — `vits` reproduces the training depth to a 100% match).

```
:: resumable (+0 = already had depth, +1000 = newly generated); skips corrupt files
.venv\Scripts\python.exe make_depth.py --split C:\tmp\csgo_new_list.txt --data_dir C:\csgo_data
```
- Corrupt hdf5 (`bad symbol table node`) are **skipped + listed** at the end — delete + re-download those.
- Build the filelist with: `dir /b C:\csgo_data\*.hdf5 > C:\tmp\csgo_new_list.txt`

---

## 4. Embeddings (the important part)

`train_from_embeddings=True` → training loads `.npz` directly; the **encoder is NOT
run at train time**. So the **embedding dim decides the model**, not the config.

**RGB (12288 spatial, default `DINO_SPATIAL_POOL=4`):**
```
.venv\Scripts\python.exe pixelbc\scripts\generate_embeddings.py --game csgo --encoder_config pixelbc\configs\csgo\aim_expert_dino_vitb.yaml --data_base_path C:\csgo_data --filelist pixelbc\data\paper_data_split\csgo_dm_train_files.txt pixelbc\data\paper_data_split\csgo_dm_val_files.txt pixelbc\data\paper_data_split\csgo_dm_test_files.txt --output_dir C:\csgo_data
```

**Depth (12288 spatial) — for RGB+D:**
```
rmdir /s /q C:\csgo_data\embeddings\dino_depth\dinov2_vitb14
.venv\Scripts\python.exe pixelbc\scripts\generate_depth_embeddings.py --encoder_config pixelbc\configs\csgo\aim_expert_dino_vitb.yaml --data_base_path C:\csgo_data --filelist pixelbc\data\paper_data_split\csgo_dm_train_files.txt pixelbc\data\paper_data_split\csgo_dm_val_files.txt --output_dir C:\csgo_data --encoder_subdir dino_depth
```
- `generate_embeddings.py` is patched to be **incremental** (skips files already embedded) — safe to re-run as data arrives.
- Each 12288-d `.npz` ≈ **30 MB** (16× the old 768-d ~2 MB).
- Verify dim: `np.load(f)['embedding'].shape == (1000, 12288)`.

---

## 5. Split management (keep validation separate)

New files (`401–1000` range) are **disjoint from val/test** (val = `100, 1001–1009`;
test = `101, 1010–…`). Rule:
- **New files → TRAIN split ONLY.** Never touch val/test (keeps the 0.77 baseline comparable + no leakage).
- Append new names with a diff of `C:\csgo_data\*.hdf5` minus (train ∪ val ∪ test).
- Both repos have their own split files — if using `_alex` (via `PYTHONPATH`), sync its split too.

---

## 6. The encoder change (CLS vs spatial)

`pixelbc\models\encoders\pretrained_encoders.py`: DINO now outputs **4×4 spatial
patch tokens** (`x_norm_patchtokens` → adaptive-pool → 12288) instead of the CLS
token (768). Env toggle **`DINO_SPATIAL_POOL`**:
- `4` (default) → 12288 spatial (the aim experiment)
- `0` → 768 CLS (loads pre-existing 768-d checkpoints; `play_csgo_rgb.bat` sets this)

**Consistency:** embeddings, model, and rollout encoder must all match. Old 768
checkpoints need `DINO_SPATIAL_POOL=0`; new 12288 checkpoints use the default 4.

---

## 7. Gotchas hit (and fixes)

| symptom | cause | fix |
|---|---|---|
| `BadZipFile` on `.npz` | training read an embedding mid-write | let the regen finish before training |
| `bad symbol table node` | corrupt/truncated hdf5 | `make_depth` now skips + lists; delete + re-download |
| doubled split path | passed `pixelbc/data/paper_data_split/...` | pass just `paper_data_split/...` (code prepends the base) |
| `NCCL not built` | `run_train.py` hardcoded `DDPStrategy` | `strategy="auto"` (single-device on Windows) |
| `prefetch_factor` error | `num_workers=0` + hardcoded prefetch/persistent | conditional `_loader_kwargs` (omit when workers=0) |
| `input 12288 ≠ 768` | model built with old CLS encoder | sync spatial encoder into the running checkout; clear `__pycache__` |
| resume size mismatch `[512,1536] vs [512,12288]` | auto-resumed the OLD checkpoint | fresh `run_name=..._spatial` (own dir, keeps old baseline) |

**Install note:** `pixelbc` had a **stale copy in site-packages** shadowing the repo.
Fixed with `uv pip install -e . --no-deps` + removing the stale `site-packages\pixelbc`.
`import pixelbc` now resolves to the repo everywhere.

---

## 8. Training

Use the `_alex` bats (fresh `_spatial` run dirs, all fixes applied):
```
cd C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games_alex
train_dino_rgb.bat        :: -> aim_expert_dino_vitb_spatial_seed_0  (RGB, 12288)
train_dino_rgbd.bat       :: -> aim_expert_dino_vitb_rgbd_spatial_seed_0 (needs 12288 depth emb)
```
Defaults: `max_epochs=20`, `check_val_every_n_epoch=1` (validate every epoch to catch
the ~epoch-4 best), top-3 + last checkpoints. Do NOT use the repo's *original* bats —
they resume the old 768/1536 checkpoints and crash.

---

## 9. Key finding (why DINO struggles in CS:GO)

Per the source paper (Schäfer et al., *Visual Encoders for Data-Efficient IL in
Modern Video Games*, arXiv:2312.02312), **pre-trained DINOv2 underperforms in
CS:GO** — Table 3 CS:GO KPM: DINOv2 ViT-B/14 = **2.98**, ViT-L/14 = 5.45, vs
**end-to-end ResNet 128 +Aug = 17.40** (≈ human non-gamer 14.32). CS:GO's
low-contrast visuals starve the frozen real-world features. So the spatial-token
change is unlikely to crack aim; **the paper's CS:GO winner is end-to-end ResNet
128 + augmentation** — that's the model that actually aims/frags in this framework.
