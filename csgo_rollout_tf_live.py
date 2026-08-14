#!/usr/bin/env python
"""Live CS:GO play with the original TeaPearce TensorFlow bot (stateful Keras-2).

The shipped csgo_rollout_tensorflow.py is a stub (loads + prints shapes only).
This is a real play loop: grab the CS:GO window -> feed one raw 0-255 frame to
the STATEFUL model (keeps LSTM state between frames) -> decode the 51-dim action
head -> inject WASD + mouse + fire. Same window/key machinery as csgo_rgb_rollout.

Frames: the model has an internal Rescaling(1/255) layer, so feed RAW 0-255 BGR
(grab_window already returns BGR at the model's 150x280) -- do NOT pre-divide or
flip to RGB (verified against the offline eval: aim corr jumps to ~0.6 this way).

Prereqs: CS:GO (legacy) running on a map, focused; tf_keras installed.
Run:
  set TF_USE_LEGACY_KERAS=1
  python csgo_rollout_tf_live.py --model trained_models/ak47_sub_55k_drop_d4_dmexpert_28_stateful --fps 16
"""
import argparse
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mss
import numpy as np
import pywintypes
import tf_keras as keras
import win32con
import win32gui

from pixelbc.data.csgo_actions import (
    CSGO_MOUSE_X_BIN_CENTERS,
    CSGO_MOUSE_Y_BIN_CENTERS,
    CSGO_N_CLICKS,
    CSGO_N_KEYS,
    CSGO_N_MOUSE_X,
)
from pixelbc.online_rollout.csgo_utils.key_input import mouse_check
from pixelbc.online_rollout.csgo_utils.key_output import (
    HoldKey,
    ReleaseKey,
    a_char,
    d_char,
    hold_left_click,
    release_left_click,
    s_char,
    set_pos,
    shift_char,
    space_char,
    w_char,
)
from pixelbc.online_rollout.csgo_utils.screen_input import grab_window
from pixelbc.online_rollout.csgo_utils.kill_tracker import KillTracker

# Windows blocks background SetForegroundWindow (focus-steal) -> make it non-fatal.
_ORIG_SFW = win32gui.SetForegroundWindow


def _safe_sfw(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        return _ORIG_SFW(hwnd)
    except pywintypes.error:
        return 0


win32gui.SetForegroundWindow = _safe_sfw

# Model key order: w,s,a,d,space,ctrl,shift,1,2,3,r. Inject movement only.
MOVE_KEYS = {"w": (0, w_char), "s": (1, s_char), "a": (2, a_char),
             "d": (3, d_char), "space": (4, space_char), "shift": (6, shift_char)}
_MX0 = CSGO_N_KEYS + CSGO_N_CLICKS
_MY0 = _MX0 + CSGO_N_MOUSE_X
_FIRE = CSGO_N_KEYS  # left-click index


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _decode(out51):
    """51-dim head -> (mouse_dx, mouse_dy [pixels], fire, set-of-held-move-keys)."""
    keys = out51[:CSGO_N_KEYS]
    keys = np.where((keys >= 0) & (keys <= 1), keys, 1 / (1 + np.exp(-keys)))
    held = {name for name, (i, _) in MOVE_KEYS.items() if keys[i] > 0.5}
    dx = float(_softmax(out51[_MX0:_MY0]) @ CSGO_MOUSE_X_BIN_CENTERS)
    dy = float(_softmax(out51[_MY0:_MY0 + len(CSGO_MOUSE_Y_BIN_CENTERS)]) @ CSGO_MOUSE_Y_BIN_CENTERS)
    fv = out51[_FIRE]
    fire = (fv if 0 <= fv <= 1 else 1 / (1 + np.exp(-fv))) > 0.5
    return dx, dy, fire, held


def main():
    p = argparse.ArgumentParser(description="Live CS:GO play with the TF bot")
    p.add_argument("--model", default="trained_models/ak47_sub_55k_drop_d4_dmexpert_28_stateful",
                   help="path stem (no ext) of the STATEFUL .json/.h5 pair")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--mouse-scale", type=float, default=1.0,
                   help="scale injected aim deltas (lower = calmer; use ~0.5 if it over-turns/spins)")
    a = p.parse_args()

    print(f"loading {a.model} ...")
    model = keras.models.model_from_json(open(a.model + ".json").read())
    model.load_weights(a.model + ".h5")
    model.reset_states()
    print(f"  in={model.input_shape} out={model.output_shape}")

    print("[tf-rollout] looking for CS:GO window...")
    hwin = 0
    while hwin == 0:
        hwin = win32gui.FindWindow(None, "Counter-Strike: Global Offensive - Direct3d 9")
        if hwin == 0:
            time.sleep(1)
    win32gui.SetForegroundWindow(hwin)
    time.sleep(1)
    sct = mss.mss()
    mon = sct.monitors[2] if len(sct.monitors) == 3 else sct.monitors[1]
    Wd, Hd = mon["width"], mon["height"]
    mid_x, mid_y = mouse_check()

    keys_down = set()
    firing = False
    kills = KillTracker()
    print("[tf-rollout] PLAY loop running. Focus CS:GO. Ctrl-C to stop.", flush=True)
    try:
        while True:
            t0 = time.time()
            raw = grab_window(hwin, game_resolution=(1024, 768), SHOW_IMAGE=False)  # BGR 150x280
            x = raw.astype(np.float32)[None, None]  # (1,1,150,280,3), raw 0-255
            out = model.predict(x, verbose=0)[0, 0]  # (52,)
            dx, dy, fire, held = _decode(out)

            for k in held - keys_down:
                HoldKey(MOVE_KEYS[k][1])
            for k in keys_down - held:
                ReleaseKey(MOVE_KEYS[k][1])
            keys_down = held
            set_pos(mid_x + dx * a.mouse_scale, mid_y + dy * a.mouse_scale, Wd, Hd)
            if fire and not firing:
                hold_left_click(); firing = True
            elif not fire and firing:
                release_left_click(); firing = False

            sk = kills.update()
            print(f"\rkeys: {' '.join(sorted(held)) or '(none)':16} aim:({dx:+5.0f},{dy:+5.0f}) "
                  f"{'FIRE' if fire else '    '} kills:{sk:>2}", end="", flush=True)
            dt = time.time() - t0
            if dt < 1 / a.fps:
                time.sleep(1 / a.fps - dt)
    finally:
        for k in list(keys_down):
            ReleaseKey(MOVE_KEYS[k][1])
        if firing:
            release_left_click()
        kills.report("evaluation/kills_tf.json")
        print("\n[tf-rollout] stopped, keys released.")


if __name__ == "__main__":
    main()
