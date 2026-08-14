#!/usr/bin/env python
"""Generate REAL per-frame depth for CS:GO HDF5 files using Depth-Anything-V2
and inject it as frame_{i}_depth. Replaces the old setup_rgb_depth.py hack
(which put one unrelated demo depth video into a single file).

Usage:
  python make_depth.py --split paper_data_split/csgo_dm_val_files.txt
  python make_depth.py --split paper_data_split/csgo_dm_train_files.txt --encoder vitl

Runs on GPU. Resumable: files/frames that already have depth are skipped.
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch

DA_ROOT = r"C:\workspace\NAVIGATION\Depth-Anything-V2"
sys.path.insert(0, DA_ROOT)
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
NUM_STEPS_PER_FILE = 1000
DEPTH_H, DEPTH_W = 150, 280  # match frame_{i}_x and existing depth shape


def load_model(encoder, device):
    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt = os.path.join(DA_ROOT, "checkpoints", f"depth_anything_v2_{encoder}.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def depth_to_uint8(depth):
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo + 1e-8) * 255.0
    return d.astype(np.uint8)


def process_file(model, hdf5_path, input_size, device, overwrite=False):
    added = 0
    with h5py.File(hdf5_path, "r+") as f:
        for i in range(NUM_STEPS_PER_FILE):
            img_key, depth_key = f"frame_{i}_x", f"frame_{i}_depth"
            if img_key not in f:
                break
            if depth_key in f:
                if not overwrite:
                    continue  # resumable
                del f[depth_key]  # replace stale/fake depth
            bgr = f[img_key][:].astype(np.uint8)  # (150,280,3), BGR as stored
            with torch.no_grad():
                depth = model.infer_image(bgr, input_size)  # HxW float, model resizes internally
            d = cv2.resize(depth_to_uint8(depth), (DEPTH_W, DEPTH_H))
            f.create_dataset(depth_key, data=d, compression="gzip")
            added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="path to split .txt (one hdf5 basename per line)")
    ap.add_argument("--data_dir", default=r"C:\csgo_data")
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--overwrite", action="store_true", help="replace existing depth (e.g. old fake depth)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | encoder: {args.encoder}")
    if device == "cpu":
        print("WARNING: no CUDA -- this will be very slow.")

    split = args.split
    if not os.path.exists(split):
        # split files live under pixelbc/data/ (same as the training config resolves)
        alt = os.path.join(os.path.dirname(__file__), "pixelbc", "data", args.split)
        if os.path.exists(alt):
            split = alt
        else:
            raise FileNotFoundError(f"Split not found: {args.split} (also tried {alt})")
    with open(split) as fh:
        files = [ln.strip() for ln in fh if ln.strip()]
    print(f"{len(files)} files from {split}")

    model = load_model(args.encoder, device)
    corrupt = []
    for n, name in enumerate(files, 1):
        p = os.path.join(args.data_dir, name)
        if not os.path.exists(p):
            print(f"  [{n}/{len(files)}] MISSING {name}")
            continue
        # A single corrupt/truncated hdf5 (bad symbol-table node) must not kill the
        # whole run -- skip it, record it, keep going.
        try:
            added = process_file(model, p, args.input_size, device, overwrite=args.overwrite)
            print(f"  [{n}/{len(files)}] {name}: +{added} depth frames")
        except Exception as exc:
            corrupt.append(name)
            print(f"  [{n}/{len(files)}] CORRUPT/SKIP {name}: {type(exc).__name__}: {exc}")
    print("Done.")
    if corrupt:
        print(f"\n{len(corrupt)} corrupt file(s) skipped (re-download these):")
        for name in corrupt:
            print(f"  {name}")


if __name__ == "__main__":
    main()
