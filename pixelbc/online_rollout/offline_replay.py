"""Offline replay: run a trained DINO model (RGB or RGB+D) over RECORDED CS:GO
frames and compare predicted vs ground-truth movement (WASD), mouse aim, and fire
-- no game needed.

Feeds ``frame_{i}_x`` (and, for the RGB+D model, the stored ``frame_{i}_depth``)
through the SAME action path the live rollouts use, one frame at a time so the LSTM
carries temporal state, then decodes the ground truth from ``frame_{i}_y``.

- RGB model  : live DINOv2 encode of the RGB frame (``model.act``), same as csgo_rgb_rollout.py.
- RGB+D model: dual embedding = DINOv2(rgb) concat DINOv2(depth), fed to the policy head,
  same as csgo_depth_rollout.py -- EXCEPT depth comes from the hdf5's stored
  ``frame_{i}_depth`` (the exact vits depth the model trained on), so no Depth-Anything
  is needed and there is zero depth distribution shift.

The model type (RGB vs RGB+D) is auto-detected from the checkpoint's embedding dim.
This tests the rollout's action path offline; it does NOT press keys or need CS:GO.

Run (GPU often busy -> --device cpu is safe):
  python pixelbc/online_rollout/offline_replay.py \
      --checkpoint training_logs/aim_expert_dino_vitb_seed_0/models/last.ckpt --num_frames 1000 --device cpu
  python pixelbc/online_rollout/offline_replay.py \
      --checkpoint training_logs/aim_expert_dino_vitb_rgbd_seed_0/models/last.ckpt --num_frames 1000 --device cpu
"""
import argparse

import h5py
import numpy as np
import torch
from torch.distributions import Bernoulli

from pixelbc.data.csgo_actions import CSGO_KEY_NAMES, preprocess_csgo_action_line
from pixelbc.data.image_preprocessing import \
    get_preprocessing_function_and_image_shape
from pixelbc.utils.load_checkpoint import load_checkpoint

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the OmegaConf
# objects in these checkpoints' hparams. Locally trained (trusted) -> force False
# (same shim as the live rollouts).
_ORIG_TORCH_LOAD = torch.load


def _load_full(*a, **k):
    k["weights_only"] = False
    return _ORIG_TORCH_LOAD(*a, **k)


torch.load = _load_full

# Movement keys we score (subset of CSGO_KEY_NAMES; 1/2/3/r weapon keys are not
# movement so they are excluded from the summary).
MOVE_KEYS = ["w", "s", "a", "d", "space", "shift"]


def _num_frames_in(h5):
    n = 0
    while f"frame_{n}_x" in h5:
        n += 1
    return n


def _predict_rgb(model, rgb_tensor):
    """RGB model: model.act encodes the frame live (train_from_embeddings=False)."""
    with torch.no_grad():
        j, tr, b = model.act(
            rgb_tensor,
            joystick_mode="deterministic",
            trigger_mode="deterministic",
            button_mode="deterministic",
        )
    j = j.cpu().numpy()
    tr = None if tr is None else tr.cpu().numpy()
    b = b.cpu().numpy()
    return j, tr, b


def _predict_depth(model, rgb_tensor, depth_tensor):
    """RGB+D model: dual embedding = DINOv2(rgb) concat DINOv2(depth) -> policy head.
    Mirrors csgo_depth_rollout._get_actions, deterministic decode."""
    enc = model.image_encoder
    with torch.no_grad():
        rgb_emb = enc(rgb_tensor.reshape(1, 1, *rgb_tensor.shape)).reshape(1, -1)
        depth_emb = enc(depth_tensor.reshape(1, 1, *depth_tensor.shape)).reshape(1, -1)
        emb = torch.cat([rgb_emb, depth_emb], dim=-1)  # (1, 1536)
        logits = model.bc_model(emb.reshape(1, 1, -1), rollout=True)
        j_logits, t_logits, b_logits = model._split_action_logits(logits.reshape(1, -1))
    j = torch.tanh(j_logits).squeeze(0).cpu().numpy()
    tr = None
    if t_logits is not None:
        tr = Bernoulli(logits=t_logits.squeeze(0)).probs.round().int().cpu().numpy()
    b = Bernoulli(logits=b_logits.squeeze(0)).probs.round().int().cpu().numpy()
    return j, tr, b


def main():
    p = argparse.ArgumentParser(description="Offline DINO replay vs recorded CS:GO actions (RGB or RGB+D)")
    p.add_argument("-ckpt", "--checkpoint", required=True)
    p.add_argument("--hdf5", default=r"C:/csgo_data/hdf5_dm_july2021_101.hdf5",
                   help="recorded CS:GO hdf5 to replay (default: a test-split file)")
    p.add_argument("--split", default=None,
                   help="path to a split .txt; aggregate over ALL its files (overrides --hdf5, --start)")
    p.add_argument("--data_dir", default=r"C:/csgo_data", help="dir holding the split's hdf5 files")
    p.add_argument("--num_frames", type=int, default=300, help="per-file frame cap (-1 = all)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu (safe) or cuda (needs free VRAM)")
    p.add_argument("--show", type=int, default=25, help="per-frame trace lines to print")
    a = p.parse_args()

    print(f"Loading model from {a.checkpoint} ...")
    model, hp = load_checkpoint(a.checkpoint, eval_mode=True)
    model.eval()
    emb_dim = model.image_encoder.get_embedding_dim()
    is_depth = model.image_encoding_dim == 2 * emb_dim
    if not is_depth:
        assert model.image_encoding_dim == emb_dim, (
            f"Unexpected image_encoding_dim {model.image_encoding_dim} vs encoder {emb_dim}"
        )
        # RGB: encode frames on the fly rather than loading precomputed embeddings.
        model.train_from_embeddings = False
    mode = "RGB+D" if is_depth else "RGB"
    device = torch.device(a.device)
    model.to(device)
    model.init_for_rollout()

    preprocess, _ = get_preprocessing_function_and_image_shape(
        hp.encoder is None,
        hp.pretrained_encoder,
        hp.image_shape[1],
        hp.image_shape[2],
    )

    key_pos = {k: CSGO_KEY_NAMES.index(k) for k in MOVE_KEYS}
    tp = {k: 0 for k in MOVE_KEYS}
    gt_pos = {k: 0 for k in MOVE_KEYS}
    pred_pos = {k: 0 for k in MOVE_KEYS}
    correct = {k: 0 for k in MOVE_KEYS}
    fire_tp = 0
    fire_gt = 0
    fire_pred = 0
    fire_correct = 0
    aim_abs_err = np.zeros(2, dtype=np.float64)
    pred_aim, gt_aim_all = [], []  # collect for correlation / R^2 / active-frame MAE
    n = 0

    with h5py.File(a.hdf5, "r") as h5:
        total = _num_frames_in(h5)
        end = min(a.start + a.num_frames, total)
        print(f"model={mode}  file={a.hdf5}  frames_in_file={total}  evaluating [{a.start}:{end})  device={device}\n")
        header = f"{'frame':>6}  {'pred_keys':<20} {'gt_keys':<20} {'pred_aim':>15} {'gt_aim':>15}  fire p/gt"
        print(header)
        print("-" * len(header))
        for i in range(a.start, end):
            x = h5[f"frame_{i}_x"][:]   # (150,280,3) uint8, BGR
            y = h5[f"frame_{i}_y"][:]   # (51,) float64
            frame = preprocess(x[:, :, ::-1])  # BGR->RGB -> (C,H,W)
            rgb_t = torch.from_numpy(np.ascontiguousarray(frame)).to(device)

            if is_depth:
                d = h5[f"frame_{i}_depth"][:]  # (150,280) uint8, stored vits depth
                if d.ndim == 3:
                    d = d[:, :, 0]
                depth3 = np.repeat(d[:, :, None], 3, axis=2)  # (150,280,3) uint8
                depth_pp = preprocess(depth3)
                depth_t = torch.from_numpy(np.ascontiguousarray(depth_pp)).to(device)
                j, tr, b = _predict_depth(model, rgb_t, depth_t)
            else:
                j, tr, b = _predict_rgb(model, rgb_t)

            gt = preprocess_csgo_action_line(y, discretize_continuous_actions=False)
            gt_aim = (float(gt[2]), float(gt[3]))
            gt_fire = gt[4] > 0.5
            gt_keys = gt[5:]  # 10 keys aligned to CSGO_KEY_NAMES

            pred_fire = bool(tr is not None and len(tr) and tr[0] > 0.5)
            pred_set, gt_set = [], []
            for k in MOVE_KEYS:
                pv = b[key_pos[k]] > 0.5
                gv = gt_keys[key_pos[k]] > 0.5
                tp[k] += int(pv and gv)
                gt_pos[k] += int(gv)
                pred_pos[k] += int(pv)
                correct[k] += int(pv == gv)
                if pv:
                    pred_set.append(k)
                if gv:
                    gt_set.append(k)
            fire_tp += int(pred_fire and gt_fire)
            fire_gt += int(gt_fire)
            fire_pred += int(pred_fire)
            fire_correct += int(pred_fire == gt_fire)
            aim_abs_err += np.abs([j[2] - gt[2], j[3] - gt[3]])
            pred_aim.append([float(j[2]), float(j[3])])
            gt_aim_all.append([gt[2], gt[3]])
            n += 1

            if i - a.start < a.show:
                print(f"{i:>6}  {' '.join(pred_set) or '-':<20} {' '.join(gt_set) or '-':<20} "
                      f"{('(%+.2f,%+.2f)' % (j[2], j[3])):>15} {('(%+.2f,%+.2f)' % gt_aim):>15}  "
                      f"{'FIRE' if pred_fire else '.':>4}/{'FIRE' if gt_fire else '.'}")

    print(f"\n==== {mode} summary over {n} frames ({a.hdf5}) ====")
    print(f"{'key':>6} {'recall':>8} {'precision':>10} {'accuracy':>9} {'gt_rate':>8}")
    for k in MOVE_KEYS:
        rec = tp[k] / gt_pos[k] if gt_pos[k] else float("nan")
        prec = tp[k] / pred_pos[k] if pred_pos[k] else float("nan")
        acc = correct[k] / n if n else float("nan")
        print(f"{k:>6} {rec:>8.3f} {prec:>10.3f} {acc:>9.3f} {gt_pos[k] / n:>8.3f}")
    fire_rec = fire_tp / fire_gt if fire_gt else float("nan")
    print(f"\nfire: recall {fire_rec:.3f}  accuracy {fire_correct / n:.3f}  gt_rate {fire_gt / n:.3f}  pred_rate {fire_pred / n:.3f}")
    print(f"aim MAE (x,y): ({aim_abs_err[0] / n:.3f}, {aim_abs_err[1] / n:.3f})  [normalized -1..1 units]")

    # Mouse/aim quality: MAE alone flatters idle prediction (aim is ~0 most frames),
    # so also report correlation, R^2, movement magnitude, and MAE on ACTIVE frames.
    pa = np.asarray(pred_aim)
    ga = np.asarray(gt_aim_all)
    print("\n---- mouse/aim detail (x, y) ----")
    for ax, lbl in ((0, "x"), (1, "y")):
        pcol, gcol = pa[:, ax], ga[:, ax]
        denom = gcol.std() * pcol.std()
        corr = float(np.corrcoef(pcol, gcol)[0, 1]) if denom > 1e-9 else float("nan")
        ss_res = float(np.sum((gcol - pcol) ** 2))
        ss_tot = float(np.sum((gcol - gcol.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
        active = np.abs(gcol) > 0.1  # frames where the human is actively turning
        mae_act = float(np.mean(np.abs(pcol[active] - gcol[active]))) if active.any() else float("nan")
        print(f"  {lbl}: corr={corr:+.3f}  R^2={r2:+.3f}  gt_std={gcol.std():.3f}  pred_std={pcol.std():.3f}  "
              f"active_frac={active.mean():.2f}  MAE_active={mae_act:.3f}")
    print("  corr/R^2 ~0 => aim not tracked; pred_std << gt_std => model barely moves the mouse")
    print("\nrecall  = of the frames the human held this key, how often the model also predicts it")
    print("gt_rate = fraction of frames the human held this key (class balance)")


if __name__ == "__main__":
    main()
