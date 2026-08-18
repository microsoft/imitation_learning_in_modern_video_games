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

## Auxiliary-prediction exploration (`predict_rgb`, `predict_embedding`) — status

Explored adding an auxiliary reconstruction/prediction task on top of the frozen-DINO
embedding (does the embedding retain enough info to reconstruct the frame, or predict
the next-step embedding). Two flavors discussed:

- **`predict_rgb`** (reconstruct current-frame RGB pixels from the embedding via a
  conv-transpose decoder, `RGBDecoder`): implemented once (wiring in `bc_model.py`,
  `data_parsers.py`, `loss_utils.py`, plus `pixelbc/models/rgb_decoder.py` and two
  configs — `aim_expert_dino_vitb_rgb_aux.yaml`, `aim_expert_dino_vitb_rgbd_rgb_aux.yaml`),
  then **reverted** in `bc_model.py`/`data_parsers.py`/`loss_utils.py` back to original.
  `rgb_decoder.py` and both `*_rgb_aux.yaml` configs are still on disk but **currently
  non-functional** — `predict_rgb`/`rgb_loss_weight` land in `BCModel`'s `**kwargs` and
  are silently ignored (no error, just a no-op) since the wiring that reads them was
  reverted.
- **`predict_embedding`** (predict next-step embedding from current embedding+action,
  JEPA-style): only ever designed/discussed, **never implemented**. The one piece that
  does exist: `get_embedding_prediction_metrics()` in
  `pixelbc/models/utils/train_metrics.py` (MSE + cosine similarity between predicted
  and target embeddings) — standalone, not called from anywhere, not logged to
  TensorBoard, waiting for the actual prediction head + `_step()` wiring.

**Checked whether this already existed elsewhere first** (before writing new code) —
searched every branch (`main`, `depth`, `predict_depth`, `dpredict`, `train`) for
embedding-prediction-with-accuracy: none exists. The `predict_depth`/`dpredict` branch
names refer to **generating depth embeddings** (`generate_depth_embeddings.py`) for the
existing RGB+D concat pipeline (§ above), not to a next-embedding-prediction auxiliary
task — its `eval_pretrained.py` "accuracy" is action accuracy (button prediction), same
metric as everywhere else in this repo, not an embedding-prediction metric.

**What does still work unconditionally, from this session:**
`data.require_depth_embedding: false` (new opt-in flag on `CSGOEmbeddingParser`) — a
file missing its depth embedding falls back to zeros for that file (warns) instead of
crashing the whole run. See `aim_expert_dino_vitb_rgbd.yaml` and the new
`train_dino_rgbd_optional_depth.bat`.

## `predict_embedding` — REMOVED from the codebase (decision: not pursuing it)

**Status update, after the v2 section below**: decided not to pursue this task further.
All code removed — `embedding_predictor.py` deleted, `bc_model.py`/`loss_utils.py`/
`train_metrics.py` wiring stripped out, `aim_expert_dino_vitb_predict_embedding.yaml`
and `train_dino_predict_embedding.bat`/`.sh` deleted. `predict_depth` (separate task,
still in the codebase) reused the same LSTM-hidden-state-branching lesson from this
work, so the investment wasn't wasted even though this specific task was dropped.
Existing `training_logs/aim_expert_dino_vitb_predict_embedding_seed_0` (the v1 run)
was also deleted on request; the v2 fix was written but never actually trained
against real data before the decision to drop it. The sections below are kept as a
historical record of what was tried and the lesson learned (branch auxiliary losses
off a shared/trainable representation, not a frozen one) — not a description of
anything currently in the repo.

## `predict_embedding` — actually built and trained this time

Unlike `predict_rgb` above, `predict_embedding` got a full second implementation pass
that stuck: `pixelbc/models/embedding_predictor.py` (`EmbeddingPredictor`, a 3-layer MLP
predicting `embedding[t+1]` from `embedding[t] + action[t]`, JEPA-style), wired into
`bc_model.py` (`predict_embedding`/`embedding_loss_weight` config flags, loss + metrics
logged every `_step()`), `compute_embedding_loss` in `loss_utils.py`, and
`aim_expert_dino_vitb_predict_embedding.yaml`. Needs no new data loading — reuses the
embedding sequence already in the batch (`train_from_embeddings=True` only).
Training scripts: `train_dino_predict_embedding.bat` / `.sh`.

### Live comparison vs. the existing RGB / RGB+D DINO baselines

Ran it against the pre-existing `aim_expert_dino_vitb_seed_0` (RGB DINO) and
`aim_expert_dino_vitb_rgbd_seed_0` (RGB+D DINO) runs' saved TensorBoard logs. Full
averages (RGB DINO n=9, RGB+D DINO n=10, predict_embedding n=14 validation checkpoints
as of this writing):

| Metric | RGB DINO | RGB+D DINO | RGB DINO + predict_embedding |
|---|---|---|---|
| `button_ind_balanced_accuracy` | 0.7756 | **0.7839** | 0.7760 |
| `button_ind_tp_rate` (button recall) | 0.5892 | **0.6077** | 0.5911 |
| `trigger_accuracy` (shoot, majority-class-dominated) | 0.9543 | **0.9574** | 0.9562 |
| `trigger_fp_rate` (false-shoot rate, lower=better) | 0.0150 | **0.0112** | 0.0120 |
| `embedding_cosine_similarity` (aux task's own signal) | n/a | n/a | ~0.884 (learning, not still climbing) |

**Result: no measurable effect either way.** `predict_embedding` tracks RGB DINO within
noise on every metric checked (balanced-acc, recall, trigger-accuracy, trigger-FP-rate) —
consistently ~0.8 points behind RGB+D DINO on all of them, same gap RGB DINO has on its
own. Depth remains the only intervention in this repo that's actually moved the needle;
the auxiliary next-embedding-prediction task doesn't appear to help the policy learn
better representations, at least not through this simple MLP-predictor formulation.

### `trigger_recall_epoch` fix (real shoot-recall, not `nan`)

`trigger_tp_rate`/`trigger_balanced_accuracy` have always logged `nan` for all three
runs — root cause: `get_trigger_metrics` computes TP-rate **per batch**
(`TP.sum() / target.sum()`), and since "shoot" is only ~4% of frames, many individual
batches have zero shoot-target frames at all → that batch's ratio is `0/0` → `nan` →
poisons the whole epoch's Lightning-averaged value even though other batches had real
positives. Fixed by accumulating TP-count and positive-count across the **entire**
validation epoch before dividing (`_trigger_epoch_stats` + `on_validation_epoch_start`/
`_end` hooks in `bc_model.py`), logged as a new metric: `{split}/trigger_recall_epoch`.
Smoke-tested standalone (correct aggregate, skips only if the whole epoch is genuinely
positive-free).

**Not yet populated for any run** — editing `bc_model.py` doesn't affect an
already-running training process (module already imported into that interpreter);
needs either (a) the training process actually killed and relaunched, or (b) retroactive
recompute via `pixelbc/validate.py <config> checkpoint=<path/to/last.ckpt> ...` against a
saved checkpoint (this script spawns a fresh process each time, so it does pick up the
fix).

**Retroactively recomputed via (b) for all three runs' `last.ckpt`** — real shoot-recall
numbers, plus a full re-check of every other metric on the same up-to-date checkpoints:

| Metric | RGB DINO | RGB+D DINO | RGB DINO + predict_embedding |
|---|---|---|---|
| `button_ind_balanced_accuracy` | 0.7701 | **0.7783** | 0.7779 |
| `button_ind_tp_rate` (button recall) | 0.5754 | **0.5960** | 0.5937 |
| `trigger_accuracy` | 0.9552 | **0.9580** | 0.9570 |
| `trigger_fp_rate` (lower=better) | 0.0134 | **0.0106** | 0.0113 |
| `trigger_recall_epoch` (real shoot recall) | 0.1756 | **0.1826** | 0.1739 |

Real shoot-recall is low across every model (~17-18%) — confirms fire is the hardest
action to learn (as `FINDINGS.md`'s "Levers to push past 0.77" section already
predicted from the ~4% positive rate / `tp_rate=nan` symptom). RGB+D DINO leads on
**every single metric** checked, including this new one; `predict_embedding` stays
tied with plain RGB DINO across the board, no metric shows separation. Command used
(one line per run, swap config + checkpoint path):
```
python pixelbc\validate.py pixelbc\configs\csgo\aim_expert_dino_vitb_predict_embedding.yaml checkpoint=training_logs\aim_expert_dino_vitb_predict_embedding_seed_0\models\last.ckpt data.data_path=C:\csgo_data data.train_split_file_path=paper_data_split/csgo_dm_train_files.txt data.validation_split_file_path=paper_data_split/csgo_dm_val_files.txt trainer.accelerator=gpu trainer.devices=1
```
**Gotcha hit while running this**: omitting `data.train_split_file_path` silently falls
back to the config-default `csgo_train_files.txt` (not `csgo_dm_train_files.txt`), which
has a real UTF-8 BOM on its first line (`ef bb bf` prefix) that `data_split.py`'s
`get_file_paths_from_split_file` doesn't strip (`path.open("r")`, no `encoding=`) —
corrupts the first filename and crashes with a confusing `FileNotFoundError`. Always
pass both `train_split_file_path` and `validation_split_file_path` explicitly.

## `predict_embedding` v2 — rewired to branch off the LSTM hidden state

The v1 result above (no measurable effect vs. plain RGB DINO) had an architectural
explanation, not just "the idea doesn't work": `EmbeddingPredictor` branched off the raw
frozen embedding tensor `x`, which has no gradient history in `train_from_embeddings=True`
mode (loaded straight from `.npz`, not computed by anything differentiable). Its loss
could only update its own weights — zero mechanism to ever influence `bc_model` (the
LSTM policy), since the two shared no computation graph. Auxiliary-task learning
normally works by shaping a *shared* trunk; here the only shared component (DINO) is
frozen, so v1 was a diagnostic probe at best, never a real auxiliary task.

**Fix**: branch `EmbeddingPredictor` off the LSTM's own hidden state instead (the
representation between `self.lstm(...)` and `mlp_out`, `lstm_hidden_size`=512-dim,
already trained by the action losses) — see `pixelbc/models/utils/model_utils.py`'s
`LSTM.forward(..., return_hidden=False)` (new kwarg, returns `(action_logits, hidden)`
when True), `pixelbc/models/embedding_predictor.py` (input/target dims now independent:
512 in, 768/1536 out), and `pixelbc/models/bc_model.py` (`forward(..., return_hidden=False)`
threads it through in the **same** `self.bc_model(...)` call used for action logits —
critical, since the LSTM is stateful and a second call would double-advance
`hidden_state`/`cell_state`; `predict_embedding` now additionally asserts
`bc_policy.type == "lstm"`, the only policy type that exposes a hidden state this way).
The *prediction target* is unchanged — still the real next-step DINO embedding — only
the predictor's *input* moved from the frozen embedding to the shared hidden state.

**Verified before restarting training** (all in this session, not yet re-run against
real data):
1. Standalone `LSTM(..., return_hidden=True)` shape check + backward-compat check
   (`return_hidden=False` default unchanged).
2. Direct proof the gradient path now exists: ran `embedding_loss.backward()` on a toy
   batch and confirmed non-zero gradients on `lstm.lstm.weight_ih_l0` and
   `lstm.mlp_in.model[0].weight` — impossible in v1 by construction.
3. Full `BCModel(**real_predict_embedding_config)` → `training_step()` → `backward()`
   smoke test (DINOv2-vitb14 encoder, LSTM 512, real `predict_embedding`/
   `embedding_loss_weight` config) — ran end-to-end, `embedding_predictor.net[0].in_features
   == 527` (512 hidden + 15 actions) as expected, LSTM gradients populated.

**Needs a fresh run, not a resume** — `EmbeddingPredictor`'s input dim changed
(768/1536 → 512), so the old `aim_expert_dino_vitb_predict_embedding_seed_0` checkpoints
won't load (shape mismatch). `train_dino_predict_embedding.bat`/`.sh` now use
`run_name=aim_expert_dino_vitb_predict_embedding_v2` (new run dir, old run/checkpoints
untouched) specifically to avoid `run_train.py`'s auto-resume-from-newest-checkpoint
picking up the incompatible old `last.ckpt`.

**Open**: re-run and compare `_v2`'s validation trajectory against the v1/RGB-DINO/
RGB+D-DINO table above — this is the real test of whether the fix changes the outcome,
not yet done as of this writing.

## `predict_depth` — implemented and verified

New auxiliary task, applying the `predict_embedding` v1→v2 lesson from the start:
reconstruct the **depth map in image/pixel space** (not embedding space) for the same
frame, from the LSTM's hidden state (gradient-connected), instead of the frozen
embedding. Distinct from `depth_embedding` (concatenating a depth *embedding* as extra
policy input, used by the RGB+D baseline) — this repo's `predict_depth` runs are
RGB-only encoder, no `depth_embedding`, per explicit instruction not to combine them.

Reuses two existing pieces rather than new code: `RGBDecoder` (`pixelbc/models/
rgb_decoder.py`) is already generic via `output_channels`, so `output_channels=1`
gives a depth decoder for free — no new decoder class. And the raw-pixel-concat
data-loading trick from the reverted `predict_rgb` attempt (concatenate flattened
raw pixels onto the embedding tensor's tail in `CSGOEmbeddingParser`, so the
`(x, action_targets)` batch contract never changes) — here loading `frame_i_depth`
(written by `make_depth.py`) instead of `frame_i_x`, resized to 224×224.

**Design** (plan file: see session transcript / `.claude/plans/reactive-whistling-tower.md`
if still present): decoder takes the LSTM's hidden state (`return_hidden=True`, the
same mechanism already built for `predict_embedding` v2) as input, reconstructs the
*same*-timestep depth map (not next-step, unlike `predict_embedding` — no frame-shift
needed), `depth_loss` (MSE) added to the total loss via a new `depth_loss_weight`.

**Status: all 4 plan steps complete, verified, ready to train.**
- `pixelbc/data/data_parsers.py` (`CSGOEmbeddingParser`) — `predict_depth` param,
  raw-depth loading (`frame_{i}_depth` via `cv2.resize` to 224×224)/flatten/concat in
  `get_all_data`, wired through `get_data_parser`.
- `pixelbc/models/utils/loss_utils.py` — `compute_depth_loss` (MSE), same pattern as
  `compute_embedding_loss`.
- `pixelbc/models/bc_model.py` — `predict_depth`/`depth_loss_weight` params + assertion
  (`train_from_embeddings`, `bc_policy.type == "lstm"`, same reasoning as
  `predict_embedding`); `self.depth_decoder = RGBDecoder(self.bc_model.hidden_size,
  output_size=224, output_channels=1)` built after `self.bc_model`; `_get_embedding`
  extended to strip the trailing raw-depth payload before it reaches the LSTM;
  `_get_depth_target` helper; `_step` now requests `return_hidden=True` whenever
  *either* `predict_embedding` or `predict_depth` is on (still only one
  `self.bc_model` call), computes `depth_loss` from `self.depth_decoder(hidden_flat)`
  vs. the real depth target, logs `{split}/depth_loss`, folds into total loss via
  `depth_loss_weight`.
- New config: `aim_expert_dino_vitb_predict_depth.yaml` (RGB-only base, no
  `depth_embedding` — per instruction not to combine them).
- New training scripts: `train_dino_predict_depth.bat`/`.sh`.
- `_save_depth_sample`: saves one real-vs-reconstructed depth frame pair (side by
  side PNG) per validation epoch to `{tensorboard log_dir}/depth_samples/
  epoch_NNN.png`, for visual QA of what the decoder is actually reconstructing —
  not just the scalar `depth_loss`. Guarded to no-op if no logger is attached
  (e.g. the standalone `BCModel(...)` smoke tests above, which have no `Trainer`).

**Verified** (same style as `predict_embedding` v2):
1. Standalone `RGBDecoder(512, output_size=224, output_channels=1)` shape/range check
   — output `(N, 1, 224, 224)`, values in `[0, 1]`.
2. Full `BCModel(**real_predict_depth_config)` → `training_step()` → `backward()` on a
   mocked batch (embedding + concatenated raw-depth payload) — ran end-to-end,
   non-zero gradient confirmed on `self.bc_model.lstm.weight_ih_l0`
   (`.abs().sum() ≈ 9.54`) — same proof-of-gradient-connection used for
   `predict_embedding` v2.

**Prerequisite**: requires `make_depth.py` to have already generated `frame_i_depth`
for every file in the train/val split — no graceful-fallback for missing depth here
(unlike `require_depth_embedding`); a missing `frame_i_depth` crashes.

**Open**: not yet run against real training data — next step is an actual training run
and comparing its validation trajectory against the RGB-DINO / RGB+D-DINO /
`predict_embedding` table above.
