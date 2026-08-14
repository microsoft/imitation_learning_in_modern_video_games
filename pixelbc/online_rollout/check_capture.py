"""Verify the live CS:GO frame capture is correct BEFORE trusting a rollout.

Finds the CS:GO window, grabs a few frames with the SAME grab_window() the
rollouts use, saves them as PNGs, and prints diagnostics that catch the two
common failure modes:

  1. Wrong window size -> grab_window's hardcoded 1024x768 offsets crop garbage.
     (CS:GO must run WINDOWED at 1024x768 for the offsets to line up.)
  2. Black frames -> GDI BitBlt on the Direct3D 9 window returns all-black
     (mean ~0, std ~0). Fix: run CS:GO in windowed mode, not exclusive fullscreen.

Open the saved PNGs: they must show the actual game view (HUD, weapon, world).
If they are black, uniform, or show the wrong crop, the model is seeing garbage
and will idle no matter how good it is.

Run (with CS:GO open on a map):
  python pixelbc/online_rollout/check_capture.py --n 5 --out C:/tmp/csgo_capture
"""
import argparse
import os
import time

import cv2
import numpy as np
import win32gui

from pixelbc.online_rollout.csgo_utils.screen_input import grab_window

WINDOW_TITLES = [
    "Counter-Strike: Global Offensive - Direct3d 9",
    "Counter-Strike: Global Offensive - Direct3D 9",
    "Counter-Strike: Global Offensive",
]


def find_csgo_window():
    for title in WINDOW_TITLES:
        h = win32gui.FindWindow(None, title)
        if h:
            return h, title
    # fall back: enumerate and match a substring
    found = []
    win32gui.EnumWindows(lambda hh, _: found.append((hh, win32gui.GetWindowText(hh))), None)
    for hh, txt in found:
        if "Counter-Strike" in txt:
            return hh, txt
    return 0, None


def main():
    p = argparse.ArgumentParser(description="Check CS:GO live frame capture")
    p.add_argument("--n", type=int, default=5, help="frames to grab")
    p.add_argument("--out", default=r"C:/tmp/csgo_capture", help="dir to save PNGs")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between grabs")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    hwin, title = find_csgo_window()
    if not hwin:
        print("ERROR: CS:GO window not found. Is the game running? Titles tried:")
        for t in WINDOW_TITLES:
            print("  -", t)
        return
    rect = win32gui.GetWindowRect(hwin)
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    print(f"Found window: {title!r}  hwnd={hwin}")
    print(f"Window rect = {rect}  size = {w}x{h}")
    if (w, h) != (1024, 768):
        print(f"  !! WARNING: window is {w}x{h}, NOT 1024x768. grab_window's hardcoded")
        print("     offsets assume 1024x768 WINDOWED -> the capture crop will be wrong.")
        print("     Fix: CS:GO video settings -> Windowed, 1024x768.")

    print(f"\nGrabbing {a.n} frames -> {a.out}")
    for i in range(a.n):
        img = grab_window(hwin, game_resolution=(1024, 768), SHOW_IMAGE=False)  # BGR (H,W,3)
        mean, std, mn, mx = float(img.mean()), float(img.std()), int(img.min()), int(img.max())
        path = os.path.join(a.out, f"capture_{i}.png")
        cv2.imwrite(path, img)
        flag = ""
        if mean < 3 and std < 3:
            flag = "  <-- BLACK/EMPTY (BitBlt failed on the D3D9 window; run CS:GO WINDOWED)"
        elif std < 5:
            flag = "  <-- nearly uniform (suspicious)"
        print(f"  frame {i}: shape={img.shape} mean={mean:6.2f} std={std:6.2f} min={mn} max={mx}  {path}{flag}")
        time.sleep(a.interval)

    print("\nNow OPEN the PNGs and confirm they show the actual game view (weapon, HUD, world).")
    print("If black/uniform/wrong-crop -> the model sees garbage and will idle. Fix capture first.")


if __name__ == "__main__":
    main()
