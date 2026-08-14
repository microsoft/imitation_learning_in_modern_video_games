#!/usr/bin/env python
"""Data validation: confirm the keyboard keys are present in the CS:GO action
labels for a split. Reads the RAW frame_{i}_y (line[0:11] = the 11 keys, line[11]
= left-click/fire, line[12] = right-click) BEFORE any preprocessing that might
drop them, and reports per-key press-rates.

Usage:
  python check_keys.py --split paper_data_split/csgo_dm_train_files.txt
  python check_keys.py --split paper_data_split/csgo_dm_test_files.txt
"""
import argparse
import os

import h5py
import numpy as np

KEYS = ["w", "s", "a", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]  # frame_i_y[0:11]


def resolve_split(split):
    if os.path.exists(split):
        return split
    alt = os.path.join(os.path.dirname(__file__), "pixelbc", "data", split)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Split not found: {split} (also tried {alt})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_dir", default=r"C:\csgo_data")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files in the split")
    args = ap.parse_args()

    names = [ln.strip() for ln in open(resolve_split(args.split)) if ln.strip()]
    if args.max_files:
        names = names[: args.max_files]

    press = np.zeros(11)
    fire = rclick = total = missing = 0
    for n in names:
        p = os.path.join(args.data_dir, n)
        if not os.path.exists(p):
            missing += 1
            continue
        with h5py.File(p, "r") as h:
            for i in range(1000):
                k = f"frame_{i}_y"
                if k not in h:
                    break
                y = np.asarray(h[k][:])
                press += (y[:11] != 0)
                fire += int(y[11] != 0)
                rclick += int(y[12] != 0)
                total += 1

    print(f"files: {len(names)} (missing {missing}) | frames: {total}\n")
    print(f"  {'key':8s} {'% pressed':>10s}")
    live = 0
    for k, c in zip(KEYS, press):
        pct = 100.0 * c / total if total else 0.0
        if c > 0:
            live += 1
        print(f"  {k:8s} {pct:9.1f}%")
    print(f"  {'fire(L)':8s} {100.0*fire/total if total else 0:9.1f}%")
    print(f"  {'R-click':8s} {100.0*rclick/total if total else 0:9.1f}%")
    print(f"\n{live}/11 keyboard keys present (nonzero) -> keys ARE in the data."
          if live else "\nNO keyboard keys present in this split!")


if __name__ == "__main__":
    main()
