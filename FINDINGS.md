# CS:GO Behavior-Cloning (pixelbc) — Findings & Experiment Log

_Investigation into why trained agents "do nothing", the depth ablation, and the pretrained-encoder result. Windows / RTX 5090 / torch cu128._

## TL;DR

1. **Root bug (fixed):** the CSGO label preprocessing discarded **all keyboard keys**, so the model could never learn movement. Fixed to keep WASD+jump+etc.
2. **The models collapse to base-rate.** From-scratch CNN/LSTM arms converge to predicting each key at its marginal frequency (ignoring the image) and freeze at **`button_ind_balanced_accuracy = 0.7493`** from epoch 2 onward — identical to 4 decimals across depth, no-depth, MLP, and LSTM.
3. **Depth (as a 4th channel) does nothing** on the collapsed CNN: RGB vs RGB+D LSTM were byte-identical (0.7493 / 0.5414 / 0.0428). Not a depth effect — the model wasn't using vision at all.
4. **Why they collapse:** the data is sparse + highly continuous + low-entropy (button sparsity ~11%, action continuity 0.947, entropy 0.12), so "predict the average / repeat last action" is the easy BCE optimum.
5. **Pretrained encoder (DINOv2) breaks the collapse.** Frozen DINOv2-vitb14 + LSTM reached **~0.77** balanced-acc — first model to leave the 0.7493 base-rate (a *distinct*, vision-conditioned number). It plateaus at ~0.77 (data-limited; the eval doc's *expected* button range is 65–80%), so it's a real but modest gain.
6. **Depth HELPS — but only on a vision-using model.** RGB+D DINO (frozen DINOv2 on RGB *and* depth, concatenated → 1536-d) beat RGB DINO in **8/10 validations**: **+1.5% balanced-acc** (0.7701→0.7815 @ep50) and **+5.4% recall** (tp 0.5754→0.6064). The value of the extra modality is *gated by whether the base model uses vision at all* — the collapsed CNN got zero from depth; DINO gets a consistent boost.
7. **Live rollout is real-time.** `csgo_depth_rollout.py` runs the RGB+D model in CS:GO at **~24 FPS** (fits the 16-FPS play rate); RGB-only would be ~160 FPS. Validated end-to-end offline (Depth-Anything + dual embedding + LSTM), untested only against the live game.

## The root bug (keys discarded)

`pixelbc/data/csgo_actions.py :: preprocess_csgo_action_line` originally mapped every keyboard key to `None` via `CSGO_ACTION_MAP` → appended 0 → the model's button targets were all zero except mouse/shoot. Fix keeps 10 keys (w,s,a,d,space,shift,1,2,3,r) via `CSGO_KEY_INDICES`. Without this, every run produced a do-nothing / NaN-metric agent.

## The collapse (base-rate prediction)

All from-scratch arms freeze at **identical** button metrics from ~epoch 2:

| Arm | arch | depth | epochs | bal_acc | tp_rate | fp_rate |
|---|---|---|---|---|---|---|
| `cnn_lstm_rgbd` | LSTM | 4-ch | 2–20 | 0.7493 | 0.5414 | 0.0428 |
| `cnn_lstm_rgb`  | LSTM | 3-ch | 5     | 0.7493 | 0.5414 | 0.0428 |
| `default` (`_main`) | MLP | 3-ch | ~4 | 0.7540 | 0.5641 | 0.0560 |

Identical-to-4-decimals across *different models* ⇒ they produce the same predictions ⇒ input-independent base-rate. Trigger (fire) never learns (`tp_rate = nan`, ~4% of frames). Joystick/aim R² ≈ 0.

**Why:** `EVALUATION_RESULTS.md` confirms button sparsity ~11% (1 in 9 frames), action continuity **0.9474** (95% frame-to-frame identical), entropy **0.12**. On such data, guessing the marginal already scores ~0.75, so there is little gradient pressure to learn frame→action.

## Depth ablation — CNN (settled: no effect)

Clean twin configs `cnn_lstm_rgb.yaml` vs `cnn_lstm_rgbd.yaml` differ by exactly one line (`use_depth`). At epoch 5, byte-identical on every button metric. **Depth adds nothing** — because neither model uses vision. This is *"depth doesn't help this collapsed model"*, not *"depth is useless"*.

## Pretrained encoder — DINOv2 (breaks collapse, then plateaus)

Per the paper (this repo = "Visual Encoders for Data-Efficient IL"), the recommendation for small data is a **pretrained** encoder, not from-scratch. Config `aim_expert_dino_vitb.yaml`: frozen DINOv2-vitb14 + LSTM, `train_from_embeddings: True`, 224×224.

RGB DINO validation trend:

| epoch | bal_acc | tp_rate | fp_rate | val/loss |
|---|---|---|---|---|
| 5 | **0.7717** | 0.5812 | 0.0379 | 1.0105 |
| 10 | 0.7704 | 0.5798 | 0.0390 | 1.1359 |

**Read:** first model to leave the 0.7493 base-rate (it conditions on the image), but only ~+0.02 and it plateaus by epoch 5 with rising val loss. Good features help a little; the data ceiling is ~0.77.

## RGB+D DINO — depth helps (the clean answer)

Dual-embedding: frozen DINOv2 on RGB **and** on depth (replicated 1→3ch), concatenated per frame → **1536-d** → LSTM. Only difference from RGB DINO = the depth stream (clean, one-variable), gated behind a `depth_embedding` flag.

Both ran to epoch 50. RGB+D beat RGB in **8/10 validations** (mean Δbalanced-acc **+0.009**):

| model | depth | bal_acc @50 | tp_rate @50 | fp_rate @50 |
|---|---|---|---|---|
| DINO LSTM | RGB   | 0.7701 | 0.5754 | 0.0353 |
| DINO LSTM | RGB+D | **0.7815** | **0.6064** | 0.0433 |

**Relative:** +1.5% balanced-acc, **+5.4% recall** (RGB+D presses the right keys more often), small fp-rate cost. Depth = ~35% of the total gain over the collapsed baseline (pretrained encoder = ~65%).

**The headline finding:** depth's usefulness is *conditional on the base model using vision*. On the collapsed CNN, RGB and RGB+D were byte-identical (0.7493) — depth added nothing. On DINO, depth adds a consistent boost. **An extra modality only helps once the model uses the modality it augments.**

## Live rollout (RGB+D, real-time)

`pixelbc/online_rollout/csgo_depth_rollout.py` — subclasses `CSGORollout`, reuses the model's **own** frozen DINOv2 for both streams, adds Depth-Anything-V2-vits, injects the 1536-d dual embedding into the LSTM (bypasses the `act()` "can't act from embeddings" assert). Validated offline: correct shapes/logits, and depth preprocessing matched **exactly** to `make_depth.py` (feed **BGR** to Depth-Anything, min-max→uint8).

- Warm latency **41.5 ms/frame ≈ 24 FPS** (RGB+D); RGB-only ~6 ms ≈ 160 FPS. Both exceed the 16-FPS CS:GO play rate.
- Requires CS:GO (legacy) + aim map; Depth-Anything-V2 checkout on path. Untested only against the live game (frame capture / control output / actual play quality).
```
python pixelbc\online_rollout\csgo_depth_rollout.py --checkpoint training_logs\aim_expert_dino_vitb_rgbd_seed_0\models\last.ckpt --fps 16 --depth_anything_dir C:\workspace\NAVIGATION\Depth-Anything-V2
```

**CS:GO setup (one-time):**
1. **Steam → legacy build:** right-click Counter-Strike 2 → Properties → **Game Versions & Betas** → Beta Participation → **`csgo_legacy`** (downloads legacy CS:GO — the version the model targets, not CS2).
2. **Steam → aim map:** subscribe to workshop id **368026786** (https://steamcommunity.com/sharedfiles/filedetails/?id=368026786).
3. **GSI config (kill/score tracking):** copy
   `pixelbc\online_rollout\csgo_utils\gamestate_integration_bcfrompixels.cfg`
   → `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`
   (note the CS2 `game\csgo\cfg` subfolder, not plain `csgo\cfg`). Optional — only for metrics.
4. **In-game:** Settings → Game → **Enable Developer Console** = Yes.

**Which map:** the model was trained on **Dust2 deathmatch** (`hdf5_dm_*`, Map: Dust2). So:
- *Quick "does it move?" test* → in-game **Play → Deathmatch → Dust II** (training-matched; not the aim map).
- *Paper-style eval* → the custom aim map via **console** (`~`): `map aimtraindriving_v3 custom` (needs workshop 368026786 subscribed). It is **not** in the Play menu.

**Run procedure:**
1. Launch CS:GO (Play), get into a map (Deathmatch→Dust II, or console `map aimtraindriving_v3 custom`).
2. Bring CS:GO to the foreground (the rollout captures the visible screen), then run the command above from the `-rgbd` repo root in a separate terminal.
3. First frame warms up (~0.1 s RGB / ~0.4 s RGB+D), then hits the FPS above. Useful flags: `--button_action_mode deterministic` (steadier keys), `--ignore_keyboard_inputs` (your typing won't interfere).

**Expected play quality** (from the model's val metrics): movement (WASD) works, decent key recall (~0.61 tp), **no firing** (trigger never learned), rough aim (joystick R²≈0).

**Both rollouts exist and are validated offline** (stock `rollout.py` can't play either DINO model — both trained `train_from_embeddings=True`, and `act()` asserts it's False):
- **RGB:** `pixelbc/online_rollout/csgo_rgb_rollout.py` — flips `train_from_embeddings=False` so the model encodes DINOv2 live on RGB frames. ~155 FPS. `--checkpoint training_logs/aim_expert_dino_vitb_seed_0/models/last.ckpt --fps 16`
- **RGB+D:** `pixelbc/online_rollout/csgo_depth_rollout.py` — same, plus Depth-Anything + dual embedding. ~24 FPS.
Both produce valid action logits offline (joystick 4 / trigger 1 / button 10); only the live game (capture + key/mouse injection) is untested. Symmetric one-command A/B in-game.

## Depth generation & assumptions

**How the depth was made** (`make_depth.py`, injected as `frame_{i}_depth` into each hdf5):
- **Model:** Depth-Anything-V2, encoder **`vits`** (default; `vitb`/`vitl` selectable via `--encoder`), `input_size=518` (the model resizes internally).
- **Input:** the stored **BGR** frame `frame_{i}_x` (150×280×3) fed **directly** to `infer_image(bgr, 518)` — *not* converted to RGB.
- **Normalization:** per-frame **min-max to 0–255 uint8** (`(d-min)/(max-min+1e-8)*255`).
- **Output:** `cv2.resize` to **150×280**, stored gzip-compressed as `frame_{i}_depth`, 1000/file. Resumable (skips existing unless `--overwrite`).
- Downstream: `generate_depth_embeddings.py` reads `frame_{i}_depth`, replicates 1→3ch, resizes to 224, runs frozen **DINOv2** → 768-d, cached under `embeddings/dino_depth/dinov2_vitb14/`.

**Assumptions & caveats (important):**
1. **Relative, not metric depth.** Depth-Anything outputs relative (disparity-like) depth, and each frame is **independently** min-max normalized — so a given uint8 value is **not** comparable across frames (frame A's "128" ≠ frame B's "128" in real distance). The model only sees *within-frame* relative structure.
2. **BGR channel order is baked in.** Training depth was computed on **BGR** input (frames are stored BGR and passed unflipped). Depth-Anything was trained on RGB, so this is technically a channel swap — but it's **consistent**, so the policy learned on BGR-derived depth. **Any rollout MUST feed BGR too** (this is the bug fixed in `csgo_depth_rollout.py`).
3. **Encoder = `vits` (CONFIRMED).** Verified by re-generating depth for `frame_0` and comparing to the stored `frame_0_depth`: `vits` reproduces it to a **100% exact pixel match** (vitl differs, ~20%). `csgo_depth_rollout.py` uses `vits`, so training and rollout depth **match exactly — no distribution shift**. (Only swap the rollout encoder if you regenerate the data with a different one.)
4. **Monocular estimate, not ground truth.** It's a neural guess from a single low-res game frame — expect artifacts on HUD/UI, sky, muzzle flash, smoke. No depth sensor involved.
5. **Two resizes.** depth is 518 (infer) → 150×280 (stored) → 224 (embedding). Minor interpolation differences from a direct 518→224 path; assumed negligible.
6. **Depth ≠ a new view.** It's derived *from the same RGB frame*, so it adds geometric structure but no genuinely new observation — which is consistent with the modest (+1.5%) gain.

## Commands (copy-paste, run from the `-rgbd` repo root)

```bat
:: 0. (one-time) generate REAL depth for the data (Depth-Anything-V2 vits) -> frame_{i}_depth
python make_depth.py --split paper_data_split/csgo_dm_train_files.txt
python make_depth.py --split paper_data_split/csgo_dm_val_files.txt
python make_depth.py --split paper_data_split/csgo_dm_test_files.txt

:: 1. precompute frozen-DINOv2 embeddings (required for train_from_embeddings)
precompute_dino_embeddings.bat           :: RGB   -> embeddings\dino\dinov2_vitb14
precompute_dino_depth_embeddings.bat     :: depth -> embeddings\dino_depth\dinov2_vitb14

:: 2. train (auto-resume from newest ckpt; validation + checkpoints every 5 epochs)
train_dino_rgb.bat                       :: RGB DINO   -> aim_expert_dino_vitb_seed_0
train_dino_rgbd.bat                      :: RGB+D DINO -> aim_expert_dino_vitb_rgbd_seed_0
train_rgb.bat / train_rgbd.bat           :: CNN baselines (collapse at 0.7493)
..\imitation_learning_in_modern_video_games_main\train.bat  :: MLP baseline

:: 3. view results
.venv\Scripts\tensorboard.exe --logdir training_logs --port 6006
:: -> http://localhost:6006 , plot val/button_ind_balanced_accuracy

:: 4. PLAY in CS:GO.
::    a) Start CS:GO (legacy), enable console (Settings->Game->Developer Console).
::    b) Get into a map. Two options:
::         - training-matched "does it move?" test: Play -> Deathmatch -> Dust II
::         - paper eval (aim map): open console with `~` and run:
::               map aimtraindriving_v3 custom
::           (needs workshop 368026786 subscribed; NOT in the Play menu)
::    c) Bring CS:GO to the FOREGROUND (the rollout captures the visible window),
::       then run one of the custom scripts below from the -rgbd repo root.
::       Do NOT use stock rollout.py -- it asserts train_from_embeddings=False and
::       only injects mouse+fire (no WASD); both DINO models need these scripts.

:: RGB DINO (~160 FPS, no depth) -- start here to confirm the loop:
python pixelbc\online_rollout\csgo_rgb_rollout.py --checkpoint training_logs\aim_expert_dino_vitb_seed_0\models\last.ckpt --fps 16

:: RGB+D DINO (~24 FPS, adds Depth-Anything-V2 depth stream):
python pixelbc\online_rollout\csgo_depth_rollout.py --checkpoint training_logs\aim_expert_dino_vitb_rgbd_seed_0\models\last.ckpt --fps 16 --depth_anything_dir C:\workspace\NAVIGATION\Depth-Anything-V2

::    Runs to the 5-min time limit then saves (results under evaluation\, -p to change).
::    Stop early: Alt-Tab off CS:GO, Ctrl+C the terminal. Useful flags:
::    --button_action_mode deterministic (steadier keys), --ignore_keyboard_inputs.

:: 5. get more data (public HF, no token). N = number of 200-file chunks
download_csgo_data.bat 3                 :: 600 more files -> add to split -> re-run steps 0,1
```

Notes: run all `.bat` from a plain `cmd`/PowerShell in the repo root (they `call .venv\Scripts\activate.bat`). `make_depth.py` / rollout use the `-rgbd` `.venv` python. Windows requires `num_workers=0` (the bats set it).

## Data pipeline / how to run

Data at `C:\csgo_data` (329 used `.hdf5` = 299 train + 10 val + 20 test; 619 unused moved to `unused\`). Windows: `num_workers=0` (worker spawn re-imports TF → crash); `strategy=auto` + clear `LOCAL_RANK` (no NCCL).

Pretrained-encoder runs need **precomputed embeddings** first (`train_from_embeddings: True` loads `.npz`, does not compute on the fly):

1. `precompute_dino_embeddings.bat`        → `embeddings/dino/dinov2_vitb14/` (RGB)
2. `precompute_dino_depth_embeddings.bat`  → `embeddings/dino_depth/dinov2_vitb14/` (depth)
3. `train_dino_rgb.bat`   / `train_dino_rgbd.bat`

CNN baselines: `train_rgb.bat` (no-depth LSTM), `train_rgbd.bat` (depth LSTM), `_main\train.bat` (MLP).
Validation every 5 epochs; checkpoints every 5 + best-val (top-3) + `last.ckpt`; auto-resume from newest `.ckpt` (`resume_from: null`).

## Files created this investigation

- `pixelbc/data/csgo_actions.py` — key-fix (KEEP keys)
- `pixelbc/utils/config_utils.py` — `use_depth`→4ch image_shape (source copy)
- `pixelbc/configs/csgo/`: `cnn_lstm_rgb.yaml`, `cnn_lstm_rgbd.yaml`, `aim_expert_dino_vitb.yaml`, `aim_expert_dino_vitb_rgbd.yaml`
- `pixelbc/scripts/generate_depth_embeddings.py` — depth→3ch→frozen DINOv2
- site-packages `data_parsers.py` (RGB+D concat, gated on `depth_embedding`) + `bc_model.py` (dim ×2)
- bats: `train_rgb`, `train_rgbd`, `train_dino_rgb`, `train_dino_rgbd`, `precompute_dino_embeddings`, `precompute_dino_depth_embeddings`, `validate`, `upload_csgo_horde` / `upload_to_horde`

## Gotchas hit

- Embedding precompute `--output_dir` must be the **data path** (`C:\csgo_data`); the script appends `embeddings/<enc>` itself (passing `.../embeddings` → double dir).
- `python pixelbc/run_train.py` imports pixelbc from **site-packages** (sys.path[0] = script dir); heredoc scripts from repo cwd import **source**. Edit the copy that the run actually uses.
- Never hand-edit `.venv\Lib\site-packages\`: a deleted line broke Lightning `data_connector.py` (IndentationError killed all training).

## Levers to push past 0.77 (untried; README-derived)

- **More data** — only 299/951 files used; the rest were deleted but re-downloadable via `download_csgo_data.bat` (public HF `TeaPearce/CounterStrike_Deathmatch`, 200-file tar chunks). Biggest lever for a data-limited plateau.
- **`discretise_joystick: True`** — the paper's method for the joystick; turns aim from regression (R²≈0) into binned classification. Most targeted fix for *aim*.
- **Focal loss / positive class-weighting** on buttons+trigger — attacks the base-rate collapse directly; the only realistic way to learn **fire** (~4% positives, currently `tp_rate=nan`).
- **Bigger encoder** `dinov2-vitl14` — more capacity (needs its own embedding precompute).
- Context: the eval doc's *expected* ranges are button 65–80%, mouse 50–65%, rare actions <20% — so 0.77 button + unlearned fire is on-target, not a bug.

## Done this investigation

- Root key-fix; collapse characterised; CNN depth ablation (no effect); DINO RGB (breaks collapse); **RGB+D DINO (depth helps, +1.5% acc / +5.4% recall, 8/10 wins)**; live depth rollout built + validated offline at ~24 FPS.
- New scripts: `generate_depth_embeddings.py`, `download_csgo_data.py`/`.bat`, `csgo_depth_rollout.py`, `precompute_dino*_embeddings.bat`, `train_dino_rgb/rgbd.bat`.
- Open (needs live game): actual in-CS:GO play quality of the RGB+D agent via `csgo_depth_rollout.py`.
