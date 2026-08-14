"""Collapse diagnostic: does the button head produce frame-varying predictions,
or is it stuck near-constant (base-rate collapse)?

Loads a checkpoint, runs it over the val split, and per key reports:
  base_rate  - fraction of frames the key is actually pressed (label)
  pred_mean  - mean predicted probability
  pred_std   - std of predicted probability ACROSS frames  <-- ~0 => constant/collapsed
  on_rate    - fraction of frames predicted "on" (prob>0.5)
  corr       - Pearson corr(pred_prob, label) across frames  <-- ~0 => not conditioning on frame
"""
import sys

import numpy as np
import torch

from pixelbc.data import get_data_module
from pixelbc.data.csgo_actions import CSGO_KEY_NAMES
from pixelbc.models.bc_model import BCModel
from pixelbc.utils.config_utils import load_config_from_file_and_cli

cfg = load_config_from_file_and_cli(sys.argv)
dm = get_data_module(cfg.data, cfg.model)
dm.setup()
loader = dm.val_dataloader()

# Load from the checkpoint's OWN saved hparams (they carry the correct 4-channel
# image_shape). Do NOT override with cfg.model -- the config's use_depth->channel
# logic is currently producing 3 channels, which mismatches this 4-channel ckpt.
model = BCModel.load_from_checkpoint(cfg.checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

all_probs, all_tgts = [], []
with torch.no_grad():
    for x, action_targets in loader:
        x = x.to(device)
        _, _, button_logits = model.forward(x, rollout=False)
        probs = torch.sigmoid(button_logits).float().cpu().numpy()  # (B*seq, n_keys)
        # button targets = last n_keys columns of the action target
        n_keys = probs.shape[-1]
        tgt = action_targets.reshape(-1, action_targets.shape[-1])[:, -n_keys:].cpu().numpy()
        all_probs.append(probs.reshape(-1, n_keys))
        all_tgts.append(tgt)

P = np.concatenate(all_probs, 0)
T = np.concatenate(all_tgts, 0)
print(f"frames evaluated: {P.shape[0]}   keys: {P.shape[1]}")
print(f"{'key':>6} | {'base_rate':>9} | {'pred_mean':>9} | {'pred_std':>9} | {'on_rate':>8} | {'corr':>7}")
for i, name in enumerate(CSGO_KEY_NAMES[: P.shape[1]]):
    p, t = P[:, i], T[:, i]
    on = float((p > 0.5).mean())
    if p.std() < 1e-8 or t.std() < 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    print(f"{name:>6} | {t.mean():>9.4f} | {p.mean():>9.4f} | {p.std():>9.5f} | {on:>8.4f} | {corr:>7.3f}")

print("\nVERDICT:")
stds = P.std(0)
print(f"  mean pred_std across keys = {stds.mean():.5f}  (near 0 => constant output => base-rate collapse)")
print(f"  keys with pred_std < 0.01 (effectively constant): "
      f"{[CSGO_KEY_NAMES[i] for i in range(P.shape[1]) if stds[i] < 0.01]}")
