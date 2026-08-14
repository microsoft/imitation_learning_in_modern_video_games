#!/usr/bin/env python
"""Validate the model's input data pipeline (incl. RGB+Depth 4th channel).

Loads a file through the REAL CSGO parser using the config's use_depth flag,
confirms the model-input tensor shape/channels, and saves a visualization of
the exact channels the network receives (RGB composite + depth).

Usage:
  python check_data.py [config] [hdf5_basename] [frame_idx]
  defaults: cnn_rgbd_lean.yaml, hdf5_dm_july2021_100.hdf5, frame 400
"""
import os
import sys

# Import the INSTALLED (site-packages) pixelbc that training uses, not the local
# source checkout that would otherwise shadow it when run from this directory.
_here = os.path.abspath(os.path.dirname(__file__) or ".")
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import cv2
import numpy as np

from pixelbc.data.data_parsers import get_data_parser
from pixelbc.utils.config_utils import load_config_from_file_and_cli


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else r"pixelbc\configs\csgo\cnn_rgbd_lean.yaml"
    name = sys.argv[2] if len(sys.argv) > 2 else "hdf5_dm_july2021_100.hdf5"
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    config = load_config_from_file_and_cli(["x", config_path, "data.data_path=C:/csgo_data"])
    print("image_shape:", tuple(config.data.image_shape), "| use_depth:", config.data.get("use_depth", False))

    parser = get_data_parser(config.model, **config.data)
    inputs, actions = parser.get_all_data(os.path.join(config.data.data_path, name))
    C = inputs.shape[1]
    print(f"parser output: {tuple(inputs.shape)} {inputs.dtype}  (N, C={C}, H, W) | actions {tuple(actions.shape)}")
    assert C == config.data.image_shape[0], "channel mismatch between data and image_shape!"
    print(f"OK: data has {C} channels, matching model input.")

    frame = inputs[idx].numpy()  # (C, H, W)
    rgb = np.transpose(frame[:3], (1, 2, 0))[:, :, ::-1].copy()  # CHW RGB -> HWC BGR for cv2
    panels = [(rgb, "RGB (ch 1-3)")]
    if C >= 4:
        panels.append((cv2.applyColorMap(frame[3], cv2.COLORMAP_INFERNO), "DEPTH (ch 4)"))
    S = 3
    imgs = []
    for img, txt in panels:
        img = cv2.resize(img, (frame.shape[2] * S, frame.shape[1] * S), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        imgs.append(img)
    out = np.hstack(imgs)
    os.makedirs(r"C:\tmp", exist_ok=True)
    cv2.imwrite(r"C:\tmp\model_input.png", out)
    print(r"saved C:\tmp\model_input.png  (exact model input, resized 128x128 per channel)")


if __name__ == "__main__":
    main()
