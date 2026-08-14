"""Live CS:GO rollout for the RGB+D DINO model.

The stock rollout (`CSGORollout`) encodes only RGB and calls `model.act()`.
The RGB+D model's policy head expects a 1536-d input = concat of a frozen-DINOv2
embedding of the RGB frame AND of the depth map (Depth-Anything on the same
frame, replicated 1->3 channels). This subclass injects that dual embedding.

It reuses the model's OWN `image_encoder` (the frozen DINOv2 already loaded from
the checkpoint) for both streams, so no second encoder is loaded — only
Depth-Anything is added.

Run:
  python pixelbc/online_rollout/csgo_depth_rollout.py \
      --checkpoint training_logs/aim_expert_dino_vitb_rgbd_seed_0/models/last.ckpt \
      --fps 16 --depth_anything_dir C:/workspace/NAVIGATION/Depth-Anything-V2

NOTE: untested against a live game (needs CS:GO running). Assumptions flagged with TODO.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from pixelbc.data.csgo_actions import (CSGO_ACTIONS, CSGO_KEY_NAMES,
                                       csgo_model_output_to_csgo_action)
from pixelbc.online_rollout.csgo_rollout import CSGORollout
from pixelbc.online_rollout.csgo_utils.key_output import (
    HoldKey, ReleaseKey, a_char, d_char, s_char, set_pos, shift_char, space_char, w_char,
)
from pixelbc.online_rollout.csgo_utils.screen_input import grab_window

MOVE_KEYS = {"w": w_char, "s": s_char, "a": a_char, "d": d_char,
             "space": space_char, "shift": shift_char}

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the OmegaConf
# objects in these checkpoints' hparams. Locally trained (trusted) -> force False.
_ORIG_TORCH_LOAD = torch.load
def _load_full(*a, **k):
    k["weights_only"] = False
    return _ORIG_TORCH_LOAD(*a, **k)
torch.load = _load_full

# Windows blocks SetForegroundWindow focus-steal from a background process ->
# make it non-fatal so the rollout doesn't crash on init.
import win32con  # noqa: E402
import win32gui  # noqa: E402
import pywintypes  # noqa: E402
_ORIG_SFW = win32gui.SetForegroundWindow
def _safe_sfw(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        return _ORIG_SFW(hwnd)
    except pywintypes.error:
        return 0
win32gui.SetForegroundWindow = _safe_sfw


def load_depth_anything(depth_dir, device, encoder="vits"):
    """Load Depth-Anything-V2 (vits). Expects the Depth-Anything-V2 checkout on
    sys.path and its checkpoint under <depth_dir>/checkpoints/."""
    sys.path.insert(0, depth_dir)
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

    cfg = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])}[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt = f"{depth_dir}/checkpoints/depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


class CSGODepthRollout(CSGORollout):
    def __init__(self, *args, depth_anything_dir=None, track_kills=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert depth_anything_dir, "Pass --depth_anything_dir (the Depth-Anything-V2 checkout)."
        self.depth_model = load_depth_anything(depth_anything_dir, self.device)
        # sanity: the loaded policy expects a doubled embedding
        assert self.model.image_encoding_dim == 2 * self.model.image_encoder.get_embedding_dim(), (
            "Checkpoint is not an RGB+D (1536-d) model; use the stock rollout for RGB."
        )
        self._keys_down = set()
        # Don't block waiting for GSI: tiny timeout + a pre-seeded fake game-state so
        # the agent starts playing IMMEDIATELY. If GSI connects, real data overwrites
        # this (real kills); otherwise it plays with kills untracked (=0).
        from pixelbc.online_rollout.csgo_utils.meta_utils import server
        server.timeout = 0.001
        server.data_all = {"map": {}, "player": {"match_stats": {"kills": 0}}}
        self._server = server
        # kill tracking (opt-in via --track_kills; needs the GSI cfg installed)
        self.track_kills = track_kills
        self._kills_display = ""
        self._start_kills = None
        self._session_kills = 0
        self._kill_events = []  # (seconds_into_session, cumulative_session_kills)
        self._kills_log_path = self.save_dir / "kills_session.json"
        if track_kills:
            print("[rollout] kill tracking ON -- polling GSI at 127.0.0.1:3000 "
                  "(needs gamestate_integration_*.cfg in csgo\\cfg).", flush=True)
        print("[rollout] starting immediately (GSI optional -- kills tracked only if GSI connects).",
              flush=True)

    def _poll_kills(self, loop_start):
        """Non-blocking GSI poll: read match_stats.kills, update session count + KPM."""
        try:
            self._server.handle_request()
            kills = self._server.data_all["player"]["match_stats"]["kills"]
        except Exception:
            return
        if self._start_kills is None:
            self._start_kills = kills
        new_total = max(0, kills - self._start_kills)
        if new_total != self._session_kills:  # a kill (or reset) happened -> log the event
            self._session_kills = new_total
            self._kill_events.append((round(time.time() - loop_start, 2), new_total))
        mins = (time.time() - loop_start) / 60.0
        kpm = self._session_kills / mins if mins > 0 else 0.0
        self._kills_display = f" | kills:{self._session_kills} ({kpm:.1f}/min)"

    def _report_kills(self, loop_start):
        mins = (time.time() - loop_start) / 60.0
        kpm = self._session_kills / mins if mins > 0 else 0.0
        if self._start_kills is None:
            print("\n[rollout] no GSI data received -- kills not tracked "
                  "(is CS:GO running with the GSI cfg? did you spawn into a match?).", flush=True)
            return
        print(f"\n[rollout] session kills: {self._session_kills} in {mins:.1f} min "
              f"= {kpm:.2f} kills/min", flush=True)
        # record to a JSON file in save_dir (kills-per-minute compatible summary)
        try:
            record = {
                "checkpoint": str(getattr(self, "checkpoint_path", "")),
                "duration_min": round(mins, 3),
                "session_kills": self._session_kills,
                "kills_per_minute": round(kpm, 3),
                "kill_events": self._kill_events,  # [ [t_seconds, cumulative_kills], ... ]
            }
            with open(self._kills_log_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"[rollout] kills recorded -> {self._kills_log_path}", flush=True)
        except Exception as e:
            print(f"[rollout] WARN: could not write kills log: {e}", flush=True)

    def _dual_embedding(self, raw_frame_bgr, rgb_frame_tensor):
        """rgb_frame_tensor: preprocessed RGB (C,H,W) uint8 already on device.
        raw_frame_bgr: original captured frame (H,W,3) BGR for depth estimation."""
        enc = self.model.image_encoder
        with torch.no_grad():
            # RGB stream (same as stock)
            rgb_emb = enc(rgb_frame_tensor.reshape(1, 1, *rgb_frame_tensor.shape)).reshape(1, -1)

            # Depth stream: match make_depth.py EXACTLY -> feed the BGR frame to
            # Depth-Anything (NOT flipped to RGB), then min-max to uint8.
            depth = self.depth_model.infer_image(np.ascontiguousarray(raw_frame_bgr))  # HxW float
            lo, hi = float(depth.min()), float(depth.max())
            depth = ((depth - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)
            depth3 = np.repeat(depth[:, :, None], 3, axis=2)
            depth_pp = self.preprocessing_function(depth3)  # same resize/normalise as RGB
            depth_t = torch.from_numpy(depth_pp).to(self.device)
            depth_emb = enc(depth_t.reshape(1, 1, *depth_t.shape)).reshape(1, -1)

        return torch.cat([rgb_emb, depth_emb], dim=-1)  # (1, 1536)

    def _get_actions(self, stacked_frames_tensor):
        # stacked_frames_tensor is the preprocessed RGB frame; self._raw_frame is set below
        emb = self._dual_embedding(self._raw_frame, stacked_frames_tensor)  # (1,1536)
        m = self.model
        with torch.no_grad():
            logits = m.bc_model(emb.reshape(1, 1, -1), rollout=True)  # (1,1,num_logits)
            j_logits, t_logits, b_logits = m._split_action_logits(logits.reshape(1, -1))
        # action selection (mirrors BCModel.act; joystick continuous, buttons/trigger Bernoulli)
        joystick = torch.tanh(j_logits).squeeze(0)
        trigger = None
        if t_logits is not None:
            td = Bernoulli(logits=t_logits.squeeze(0))
            trigger = (td.sample() if self.trigger_action_mode == "stochastic" else td.probs.round()).int()
        bd = Bernoulli(logits=b_logits.squeeze(0))
        button = (bd.sample() if self.button_action_mode == "stochastic" else bd.probs.round()).int()
        button_np = button.cpu().numpy()
        # inject the movement keys the stock rollout ignores
        want = {CSGO_KEY_NAMES[i] for i in range(len(button_np))
                if button_np[i] > 0.5 and CSGO_KEY_NAMES[i] in MOVE_KEYS}
        for k in want - self._keys_down:
            HoldKey(MOVE_KEYS[k])
        for k in self._keys_down - want:
            ReleaseKey(MOVE_KEYS[k])
        self._keys_down = want
        joy = joystick.cpu().numpy()
        tr_np = None if trigger is None else trigger.cpu().numpy()
        fire = "FIRE" if (tr_np is not None and len(tr_np) and tr_np[0] > 0.5) else "    "
        print(f"\rkeys: {' '.join(sorted(want)) or '(none)':18} aim:({joy[2]:+.2f},{joy[3]:+.2f}) {fire}{self._kills_display}",
              end="", flush=True)
        return joy, tr_np, button_np

    def _process_frame(self, raw_frame):
        # stash the raw frame so _get_actions can run depth estimation on it
        self._raw_frame = raw_frame
        return super()._process_frame(raw_frame)

    def rollout_loop(self):
        """Clean play loop: grab->act->apply. Optionally poll GSI for live kills."""
        self.model.init_for_rollout()
        mode = "GSI kills ON" if self.track_kills else "no GSI/restart/video"
        print(f"[rollout] PLAY loop running ({mode}). Focus CS:GO. Ctrl-C to stop.", flush=True)
        loop_start = time.time()
        try:
            while True:
                t0 = time.time()
                if self.track_kills:
                    self._poll_kills(loop_start)
                raw = grab_window(self.hwin_csgo, game_resolution=(1024, 768), SHOW_IMAGE=False)  # BGR
                self._raw_frame = raw  # BGR for the depth path
                frame = self.preprocessing_function(raw[:, :, ::-1])  # BGR->RGB, preprocessed (C,H,W)
                tens = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
                j, tr, b = self._get_actions(tens)
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
            if self.track_kills:
                self._report_kills(loop_start)


def main():
    p = argparse.ArgumentParser(description="RGB+D DINO live CS:GO rollout")
    p.add_argument("-ckpt", "--checkpoint", required=True)
    p.add_argument("-fps", "--fps", type=int, default=16)
    p.add_argument("-p", "--path", default="evaluation")
    p.add_argument("--depth_anything_dir", required=True)
    p.add_argument("-ja", "--joystick_action_mode", default="deterministic")
    p.add_argument("-ta", "--trigger_action_mode", default="stochastic")
    p.add_argument("-ba", "--button_action_mode", default="stochastic")
    p.add_argument("--ignore_keyboard_inputs", action="store_true")
    p.add_argument("--track_kills", action="store_true",
                   help="poll GSI (127.0.0.1:3000) for live kills + kills/min")
    a = p.parse_args()
    rollout = CSGODepthRollout(
        checkpoint_path=a.checkpoint,
        save_dir=a.path,
        joystick_action_mode=a.joystick_action_mode,
        trigger_action_mode=a.trigger_action_mode,
        button_action_mode=a.button_action_mode,
        fps=a.fps,
        ignore_keyboard_inputs=a.ignore_keyboard_inputs,
        depth_anything_dir=a.depth_anything_dir,
        track_kills=a.track_kills,
    )
    rollout.rollout_loop()


if __name__ == "__main__":
    main()
