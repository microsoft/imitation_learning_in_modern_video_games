#!/usr/bin/env python
"""Check depth coverage in the CS:GO HDF5 dataset.

For each file, reports how many frames have frame_{i}_depth vs frame_{i}_x,
so you can verify depth before/after running make_depth.py.

Usage:
  python check_depth.py --split paper_data_split/csgo_dm_train_files.txt
  python check_depth.py --split paper_data_split/csgo_dm_val_files.txt
  python check_depth.py                # scan every hdf5 in data_dir
"""
import argparse
import glob
import os

import h5py


def resolve_split(split):
    if os.path.exists(split):
        return split
    alt = os.path.join(os.path.dirname(__file__), "pixelbc", "data", split)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Split not found: {split} (also tried {alt})")


def file_coverage(path):
    with h5py.File(path, "r") as f:
        keys = f.keys()
        n_rgb = sum(1 for k in keys if k.startswith("frame_") and k.endswith("_x"))
        n_depth = sum(1 for k in keys if k.startswith("frame_") and k.endswith("_depth"))
    return n_rgb, n_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="split .txt; omit to scan all files in data_dir")
    ap.add_argument("--data_dir", default=r"C:\csgo_data")
    ap.add_argument("--verbose", action="store_true", help="print every file (default: only partial/empty)")
    args = ap.parse_args()

    if args.split:
        with open(resolve_split(args.split)) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        paths = [os.path.join(args.data_dir, n) for n in names]
    else:
        paths = sorted(glob.glob(os.path.join(args.data_dir, "hdf5_dm_july2021_*.hdf5")))

    full = partial = empty = missing = 0
    frames_total = frames_depth = 0
    print(f"Checking {len(paths)} files...\n")
    for p in paths:
        name = os.path.basename(p)
        if not os.path.exists(p):
            print(f"  MISSING  {name}")
            missing += 1
            continue
        n_rgb, n_depth = file_coverage(p)
        frames_total += n_rgb
        frames_depth += n_depth
        pct = 100.0 * n_depth / n_rgb if n_rgb else 0.0
        if n_depth == 0:
            empty += 1
            status = "EMPTY"
        elif n_depth < n_rgb:
            partial += 1
            status = "PARTIAL"
        else:
            full += 1
            status = "full"
        if args.verbose or status != "full":
            print(f"  {status:8s} {name:26s} depth {n_depth}/{n_rgb} ({pct:.0f}%)")

    print("\n==== summary ====")
    print(f"  files: {len(paths)} | full={full}  partial={partial}  empty={empty}  missing={missing}")
    if frames_total:
        print(f"  frames: {frames_depth}/{frames_total} have depth ({100.0*frames_depth/frames_total:.1f}%)")


if __name__ == "__main__":
    main()
