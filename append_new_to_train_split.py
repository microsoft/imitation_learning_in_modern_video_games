"""Append any hdf5 in C:\\csgo_data that is not already in the train/val/test
splits onto the TRAIN split (never val/test -- keeps the yardstick comparable).

Run after merging new files into C:\\csgo_data:
    .venv\\Scripts\\python.exe append_new_to_train_split.py
"""
import glob
import os

DATA_DIR = r"C:\csgo_data"
# Both checkouts have their own split files -- keep them in sync so the ResNet
# (trained from _alex) and the DINO/ViT bats (-rgbd) see the same train set.
SPLIT_DIRS = [
    r"C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games-rgbd\pixelbc\data\paper_data_split",
    r"C:\workspace\NAVIGATION\imitation_learning_in_modern_video_games_alex\pixelbc\data\paper_data_split",
]


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.hdf5"))}
    for split_dir in SPLIT_DIRS:
        if not os.path.isdir(split_dir):
            print(f"skip (no dir): {split_dir}")
            continue
        train_p = os.path.join(split_dir, "csgo_dm_train_files.txt")
        train = _read(train_p)
        val = set(_read(os.path.join(split_dir, "csgo_dm_val_files.txt")))
        test = set(_read(os.path.join(split_dir, "csgo_dm_test_files.txt")))
        known = set(train) | val | test
        new = sorted(on_disk - known)
        if not new:
            print(f"[{os.path.basename(os.path.dirname(os.path.dirname(split_dir)))}] no new files "
                  f"(train {len(train)} | val {len(val)} | test {len(test)})")
            continue
        with open(train_p, "a") as f:
            for name in new:
                f.write(name + "\n")
        repo = split_dir.split("imitation_learning_in_modern_video_games")[1].split("\\")[0] or "-rgbd"
        print(f"[{repo}] appended {len(new)} NEW files to TRAIN: {len(train)} -> {len(train) + len(new)} "
              f"(val/test untouched)")


if __name__ == "__main__":
    main()
