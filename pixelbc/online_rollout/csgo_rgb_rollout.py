"""Live CS:GO rollout for the RGB DINO model (no depth).

The RGB DINO model was trained `train_from_embeddings=True` (loads cached .npz),
but live play feeds frames. This flips `train_from_embeddings=False` after load
so the model runs its own frozen DINOv2 encoder live on each RGB frame; the stock
`act()` path then works unchanged. Much simpler than the depth rollout — no
Depth-Anything, single 768-d encode (~160 FPS).

Run (with CS:GO + aim map running):
  python pixelbc/online_rollout/csgo_rgb_rollout.py \
      --checkpoint training_logs/aim_expert_dino_vitb_seed_0/models/last.ckpt --fps 16
"""
import argparse

import torch

import time

import numpy as np

from pixelbc.data.csgo_actions import (CSGO_ACTIONS, CSGO_KEY_NAMES,
                                       csgo_model_output_to_csgo_action)
from pixelbc.online_rollout.csgo_rollout import CSGORollout
from pixelbc.online_rollout.csgo_utils.key_output import (
    HoldKey, ReleaseKey, a_char, d_char, s_char, set_pos, shift_char, space_char, w_char,
)
from pixelbc.online_rollout.csgo_utils.screen_input import grab_window
from pixelbc.online_rollout.csgo_utils.kill_tracker import KillTracker

# Movement keys the stock rollout never presses (it only did mouse+fire). Maps the
# model's button outputs -> keyboard scan codes. Weapon-switch/reload (1,2,3,r) are
# intentionally left out to avoid constant gun-swapping during a movement test.
MOVE_KEYS = {"w": w_char, "s": s_char, "a": a_char, "d": d_char,
             "space": space_char, "shift": shift_char}

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the OmegaConf
# objects in these checkpoints' hparams. They are locally trained (trusted), so
# force weights_only=False for the rollout's load_checkpoint().
_ORIG_TORCH_LOAD = torch.load
def _load_full(*a, **k):
    k["weights_only"] = False
    return _ORIG_TORCH_LOAD(*a, **k)
torch.load = _load_full

# Windows blocks SetForegroundWindow focus-steal from a background process, which
# crashes the stock rollout. Make it non-fatal (restore + best-effort raise).
import win32con  # noqa: E402
import win32gui  # noqa: E402
import pywintypes  # noqa: E402
_ORIG_SFW = win32gui.SetForegroundWindow
def _safe_sfw(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        return _ORIG_SFW(hwnd)
    except pywintypes.error:
        return 0  # focus not granted; click the CS:GO window manually
win32gui.SetForegroundWindow = _safe_sfw


class CSGORgbRollout(CSGORollout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # encode RGB frames live instead of loading precomputed embeddings
        assert self.model.image_encoding_dim == self.model.image_encoder.get_embedding_dim(), (
            "This is an RGB+D (doubled-dim) checkpoint; use csgo_depth_rollout.py instead."
        )
        self.model.train_from_embeddings = False
        self._keys_down = set()
        # Don't block waiting for GSI: tiny timeout + a pre-seeded fake game-state so
        # the agent starts playing IMMEDIATELY. If GSI connects, real data overwrites
        # this (real kills); otherwise it plays with kills untracked (=0).
        from pixelbc.online_rollout.csgo_utils.meta_utils import server
        server.timeout = 0.001
        server.data_all = {"map": {}, "player": {"match_stats": {"kills": 0}}}
        print("[rollout] starting immediately (GSI optional -- kills tracked only if GSI connects).",
              flush=True)
        self._kills = KillTracker()

    def _get_actions(self, stacked_frames_tensor):
        j, tr, b = super()._get_actions(stacked_frames_tensor)
        # inject the movement keys the stock rollout ignores
        want = {CSGO_KEY_NAMES[i] for i in range(len(b))
                if b[i] > 0.5 and CSGO_KEY_NAMES[i] in MOVE_KEYS}
        for k in want - self._keys_down:
            HoldKey(MOVE_KEYS[k])
        for k in self._keys_down - want:
            ReleaseKey(MOVE_KEYS[k])
        self._keys_down = want
        fire = "FIRE" if (tr is not None and len(tr) and tr[0] > 0.5) else "    "
        sk = self._kills.update()
        print(f"\rkeys: {' '.join(sorted(want)) or '(none)':18} aim:({j[2]:+.2f},{j[3]:+.2f}) {fire} kills:{sk:>2}",
              end="", flush=True)
        return j, tr, b

    def rollout_loop(self):
        """Clean play loop: no GSI, no mp_restartgame, no video -- just grab->act->apply."""
        self.model.init_for_rollout()
        print("[rollout] PLAY loop running (no GSI/restart/video). Focus CS:GO. Ctrl-C to stop.", flush=True)
        try:
            while True:
                t0 = time.time()
                raw = grab_window(self.hwin_csgo, game_resolution=(1024, 768), SHOW_IMAGE=False)  # BGR
                self._raw_frame = raw  # BGR (used by the depth wrapper's _get_actions)
                frame = self.preprocessing_function(raw[:, :, ::-1])  # BGR->RGB, preprocessed (C,H,W)
                tens = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
                j, tr, b = self._get_actions(tens)  # injects WASD keys, prints, returns numpy actions
                act = csgo_model_output_to_csgo_action(j, tr, b)
                set_pos(self.mouse_x_mid + act[CSGO_ACTIONS.MOUSE_X],
                        self.mouse_y_mid + act[CSGO_ACTIONS.MOUSE_Y], self.Wd, self.Hd)
                dt = time.time() - t0
                if dt < 1 / 16:
                    time.sleep(1 / 16 - dt)
        finally:
            for k in list(self._keys_down):
                ReleaseKey(MOVE_KEYS[k])
            self._keys_down = set()
            self._kills.report(self.save_dir / "kills.json")


def main():
    p = argparse.ArgumentParser(description="RGB DINO live CS:GO rollout")
    p.add_argument("-ckpt", "--checkpoint", required=True)
    p.add_argument("-fps", "--fps", type=int, default=16)
    p.add_argument("-p", "--path", default="evaluation")
    p.add_argument("-ja", "--joystick_action_mode", default="deterministic")
    p.add_argument("-ta", "--trigger_action_mode", default="stochastic")
    p.add_argument("-ba", "--button_action_mode", default="stochastic")
    p.add_argument("--ignore_keyboard_inputs", action="store_true")
    a = p.parse_args()
    rollout = CSGORgbRollout(
        checkpoint_path=a.checkpoint,
        save_dir=a.path,
        joystick_action_mode=a.joystick_action_mode,
        trigger_action_mode=a.trigger_action_mode,
        button_action_mode=a.button_action_mode,
        fps=a.fps,
        ignore_keyboard_inputs=a.ignore_keyboard_inputs,
    )
    rollout.rollout_loop()


if __name__ == "__main__":
    main()
