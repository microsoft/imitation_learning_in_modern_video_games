#!/usr/bin/env python
"""Offline eval of the original TeaPearce TensorFlow bots on recorded CS:GO frames.

Loads a Keras-2 (.json arch + .h5 weights) model via tf_keras, runs it over
``frame_{i}_x`` from an hdf5 in 96-frame windows (its fixed sequence length),
decodes the 51-dim action head the SAME way as ground truth, and reports the
identical metrics as offline_replay.py (WASD recall, aim corr/R^2, fire) so the
TF bot sits directly next to the DINO numbers.

Model output layout (52 = 51 action + 1 value):
  keys[0:11]=w,s,a,d,space,ctrl,shift,1,2,3,r  clicks[11:13]=L,R
  mouse_x[13:36] (23 bins)  mouse_y[36:51] (15 bins)  value[51]

Frames are stored BGR and the model trained on them BGR -> feed as-is (no flip),
scaled to [0,1]. (If metrics come out at base-rate with ~0 correlation, that scale
assumption is the first thing to revisit.)

Run (TF is CPU-only on native Windows -> keep num_frames modest):
  set TF_USE_LEGACY_KERAS=1
  python offline_eval_tensorflow.py --model trained_models/ak47_sub_55k_drop_d4_dmexpert_28 --hdf5 C:/csgo_data/hdf5_dm_july2021_1003.hdf5 --num_frames 288
"""
import argparse
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import tf_keras as keras

from pixelbc.data.csgo_actions import (
    CSGO_MOUSE_X_BIN_CENTERS,
    CSGO_MOUSE_X_NORMALIZER,
    CSGO_MOUSE_Y_BIN_CENTERS,
    CSGO_MOUSE_Y_NORMALIZER,
    CSGO_N_CLICKS,
    CSGO_N_KEYS,
    CSGO_N_MOUSE_X,
    preprocess_csgo_action_line,
)

SEQ = 96  # model's fixed sequence length
# Model key order is w,s,a,d,space,ctrl,shift,1,2,3,r. We score movement keys,
# mapping to their index in that 11-slot vector (skip ctrl at 5).
MOVE_KEYS = {"w": 0, "s": 1, "a": 2, "d": 3, "space": 4, "shift": 6}
_MX0 = CSGO_N_KEYS + CSGO_N_CLICKS
_MY0 = _MX0 + CSGO_N_MOUSE_X
_FIRE = CSGO_N_KEYS  # left click index


def _to_probs(x):
    if np.all(x >= 0) and abs(float(x.sum()) - 1.0) < 0.05:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def _decode_pred(out51):
    """model 51-dim -> (aim_x, aim_y, fire_bool, {key: bool})."""
    keys = out51[:CSGO_N_KEYS]
    keys = np.where((keys >= 0) & (keys <= 1), keys, 1 / (1 + np.exp(-keys)))
    fire = keys_or = None
    mx = _to_probs(out51[_MX0:_MY0]) @ CSGO_MOUSE_X_BIN_CENTERS / CSGO_MOUSE_X_NORMALIZER
    my = _to_probs(out51[_MY0:_MY0 + len(CSGO_MOUSE_Y_BIN_CENTERS)]) @ CSGO_MOUSE_Y_BIN_CENTERS / CSGO_MOUSE_Y_NORMALIZER
    fv = out51[_FIRE]
    fire = (fv if 0 <= fv <= 1 else 1 / (1 + np.exp(-fv))) > 0.5
    return float(mx), float(my), bool(fire), {k: keys[i] > 0.5 for k, i in MOVE_KEYS.items()}


def _num_frames(h5):
    n = 0
    while f"frame_{n}_x" in h5:
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="Offline eval of TF CS:GO bots vs recorded actions")
    p.add_argument("--model", required=True, help="path stem (no ext) of the .json/.h5 pair")
    p.add_argument("--hdf5", default=r"C:/csgo_data/hdf5_dm_july2021_1003.hdf5")
    p.add_argument("--num_frames", type=int, default=288, help="rounded down to a multiple of 96")
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()

    arch = open(a.model + ".json").read()
    model = keras.models.model_from_json(arch)
    model.load_weights(a.model + ".h5")
    print(f"loaded {a.model}  in={model.input_shape} out={model.output_shape}")

    tp = {k: 0 for k in MOVE_KEYS}
    gt_pos = {k: 0 for k in MOVE_KEYS}
    pred_pos = {k: 0 for k in MOVE_KEYS}
    correct = {k: 0 for k in MOVE_KEYS}
    fire_tp = fire_gt = fire_pred = fire_correct = 0
    pa, ga = [], []
    n = 0

    with h5py.File(a.hdf5, "r") as h5:
        total = _num_frames(h5)
        nwin = min(a.num_frames, total - a.start) // SEQ
        print(f"model file={a.hdf5} frames={total} windows={nwin} ({nwin * SEQ} frames)\n")
        for w in range(nwin):
            s = a.start + w * SEQ
            # Model has an internal Rescaling(1/255) layer -> feed RAW 0-255, do
            # NOT pre-divide (double-scaling zeroed the input in the first pass).
            xs = np.stack([h5[f"frame_{s + t}_x"][:] for t in range(SEQ)]).astype(np.float32)
            out = model.predict(xs[None], verbose=0)[0]  # (96, 52)
            if w == 0:
                o = out[0]
                print(f"  [diag] out range [{o.min():.2f},{o.max():.2f}]  "
                      f"keys sum {o[:CSGO_N_KEYS].sum():.2f}  mx sum {o[_MX0:_MY0].sum():.2f}")
            for t in range(SEQ):
                mx, my, fire, keys = _decode_pred(out[t])
                gt = preprocess_csgo_action_line(h5[f"frame_{s + t}_y"][:], discretize_continuous_actions=False)
                gk = gt[5:]  # 10 keys w,s,a,d,space,shift,1,2,3,r
                gk_idx = {"w": 0, "s": 1, "a": 2, "d": 3, "space": 4, "shift": 5}
                for k in MOVE_KEYS:
                    pv, gv = keys[k], gk[gk_idx[k]] > 0.5
                    tp[k] += int(pv and gv); gt_pos[k] += int(gv)
                    pred_pos[k] += int(pv); correct[k] += int(pv == gv)
                gfire = gt[4] > 0.5
                fire_tp += int(fire and gfire); fire_gt += int(gfire)
                fire_pred += int(fire); fire_correct += int(fire == gfire)
                pa.append([mx, my]); ga.append([gt[2], gt[3]]); n += 1

    print(f"\n==== TF {os.path.basename(a.model)} over {n} frames ====")
    print(f"{'key':>6} {'recall':>8} {'precision':>10} {'accuracy':>9} {'gt_rate':>8}")
    for k in MOVE_KEYS:
        rec = tp[k] / gt_pos[k] if gt_pos[k] else float("nan")
        prec = tp[k] / pred_pos[k] if pred_pos[k] else float("nan")
        print(f"{k:>6} {rec:>8.3f} {prec:>10.3f} {correct[k] / n:>9.3f} {gt_pos[k] / n:>8.3f}")
    print(f"\nfire: recall {fire_tp / fire_gt if fire_gt else float('nan'):.3f}  "
          f"accuracy {fire_correct / n:.3f}  gt_rate {fire_gt / n:.3f}  pred_rate {fire_pred / n:.3f}")
    pa, ga = np.asarray(pa), np.asarray(ga)
    for ax, lbl in ((0, "x"), (1, "y")):
        pc, gc = pa[:, ax], ga[:, ax]
        corr = float(np.corrcoef(pc, gc)[0, 1]) if gc.std() * pc.std() > 1e-9 else float("nan")
        r2 = 1 - float(np.sum((gc - pc) ** 2)) / float(np.sum((gc - gc.mean()) ** 2) + 1e-9)
        act = np.abs(gc) > 0.1
        mae_act = float(np.mean(np.abs(pc[act] - gc[act]))) if act.any() else float("nan")
        print(f"aim {lbl}: corr={corr:+.3f} R^2={r2:+.3f} gt_std={gc.std():.3f} pred_std={pc.std():.3f} MAE_active={mae_act:.3f}")


if __name__ == "__main__":
    main()
