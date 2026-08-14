# CS:GO pixelbc — Script Reference

All paths relative to the repo root:
`C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games-rgbd`
Run bats from anywhere (they `cd` to their own dir). Uses the repo `.venv`.
Data lives in `C:\csgo_data` (438 hdf5, 408 train / 10 val / 20 test split).

---

## 1. Play a bot live (inference)

Prereqs for ALL: CS:GO (legacy) → **Deathmatch → Dust II**, spawn with **AK-47**,
game window **focused**, console: `m_rawinput 0; sensitivity 2.5`, res **1024x768 4:3**.

| bat | model | notes |
|---|---|---|
| `play_csgo_tf.bat July_remoterun7_g9_4k_n32_recipe_ton96__e14` | TF EfficientNet+ConvLSTM, full-dataset | **best all-rounder** |
| `play_csgo_tf.bat dmexpert_28` | TF, expert-DM finetune | **sharpest aim** (default) |
| `play_csgo_resnet.bat` | PyTorch ResNet 128 +Aug | the paper's aiming winner |
| `play_csgo_rgbd.bat` | DINO RGB+D CLS (1536) | kill tracking ON by default; `nokills` to disable |
| `play_csgo_rgb.bat` | DINO RGB CLS (768) | weak/no aim (baseline) |
| `play_csgo_rgb_spatial.bat` | DINO RGB spatial (12288) | experiment |
| `play_csgo_rgbd_spatial.bat` | DINO RGB+D spatial (24576) | experiment |

- Inference runs the encoder **live per frame** — no embeddings/data needed, just the checkpoint + the running game.
- `play_csgo_rgbd.bat` writes `evaluation\kills_session.json` (kills, kills/min, timeline) via GSI (cfg `gamestate_integration_bcfrompixels.cfg` already installed).

## 2. Train

| bat | model | data path |
|---|---|---|
| `continue_rgbd.bat` | resume DINO RGB+D (1536) from `last.ckpt` | embeddings (768 dino + 768 dino_depth) |
| `train_dino_rgb.bat` | DINO RGB (frozen encoder + LSTM) | embeddings/dino |
| `train_dino_rgbd.bat` | DINO RGB+D | embeddings/dino + dino_depth |
| `train_resnet128_aug.bat` (in `_alex`) | end-to-end ResNet 128 +Aug | **raw frames** |
| `train_vit128_aug.bat` | end-to-end ViT 128 +Aug | **raw frames** |

- **Frozen-DINO** trains (`train_dino_*`, `continue_rgbd`) read precomputed `.npz` embeddings (`train_from_embeddings=True`) — the encoder does NOT run at train time; the embedding dim must match the model.
- **End-to-end** trains (`resnet`, `vit`) read **raw 128x128 frames** (`train_from_embeddings=False`) — no embeddings needed; the CNN/ViT learns CS:GO features. **These are the models that actually aim.**
- All: `check_val_every_n_epoch=1`, `checkpoint_epoch_freq=1` → validate + checkpoint every epoch (keeps top-3 by val_loss + latest + last).
- Resume any run: `run_train.py` auto-resumes newest `*.ckpt` in the run dir, or pass `fit.ckpt_path=<path>`.

## 3. Data prep

| script | purpose |
|---|---|
| `download_csgo_new.bat <tar>` | Xet-free streaming download of a 200-file chunk into `C:\csgo_data_new` |
| `download_csgo_stream.py` | underlying streamer (`--tars`, `--dest`, `--max-files`) |
| `make_depth.py --split <list> --data_dir C:\csgo_data` | inject `frame_i_depth` via Depth-Anything-V2 vits (skips corrupt files) |
| `precompute_dino_embeddings.bat` / `generate_embeddings.py` | RGB embeddings → `embeddings/dino/dinov2_vitb14` |
| `precompute_dino_depth_embeddings.bat` / `generate_depth_embeddings.py --encoder_subdir dino_depth` | depth embeddings → `embeddings/dino_depth/dinov2_vitb14` |

- Full pipeline walkthrough + gotchas: **`DATA_PREP.md`**.
- Merge new files into `C:\csgo_data`, append new names to the **train** split only (never val/test).
- Embedding dim = CLS 768 (current committed code) unless the spatial-token encoder is active.

## 4. Upload (Horde)

Horde = remote box `10.57.233.24`, via WSL rsync (resumable — rerun to resume).

| bat | uploads |
|---|---|
| `upload_csgo_horde.bat` | `C:\csgo_data` (~45.7 GB, excludes `unused\`) → `horde:~/csgo_data/` |
| `upload_models_horde.bat` | `training_logs\` (~470 MB) → `horde:~/csgo_training_logs/` |

Enter the horde password when prompted. Interrupted? Just re-run — rsync resumes.

## 5. Key python scripts

| script | purpose |
|---|---|
| `csgo_rollout_tf_live.py` | TF bot live loop (stateful Keras, WASD/mouse/fire injection, `--mouse-scale`) |
| `pixelbc/online_rollout/csgo_rgb_rollout.py` | PyTorch RGB/ResNet/ViT live rollout |
| `pixelbc/online_rollout/csgo_depth_rollout.py` | RGB+D live rollout (Depth-Anything per frame + GSI kills) |
| `check_data.py` / `check_depth.py` / `check_keys.py` | quick data/embedding/action sanity checks |

## 6. Model roster (findings)

- **TF EfficientNet+ConvLSTM** (`July_remoterun*`, `ak47_*_dmexpert_28`): trained, **aims** (paper ~17 KPM ≈ human).
- **ResNet 128 +Aug** (PyTorch, end-to-end): the paper's CS:GO winner (17.40 KPM). **Recommended DL route.**
- **DINO (frozen) RGB / RGB+D**: offline val aim R² **negative**; does **not** aim well (paper 2.98 KPM). More data/epochs did not fix it (epoch 49 val_loss 0.68 was best; degraded after).
- Offline val R² is a weak proxy — the real metric is **in-game kills/min** (`kills_session.json`).

## 7. Gotchas

- `.bat` files need **CRLF** line endings (edit tools write LF → cmd misparses).
- `num_workers=0` on Windows (TF re-import / worker-spawn crashes) — the "many workers" warning is harmless.
- `strategy="auto"` / clear `LOCAL_RANK` — no NCCL on Windows.
- Check `nvidia-smi` before any run — shared 32 GB card.
- Frozen-DINO embeddings and the checkpoint must be the **same dim** (768 CLS vs 12288 spatial). The 12288 spatial set is backed up in `embeddings/dino*/dinov2_vitb14_spatial12288_bak` — swap folders to switch.
