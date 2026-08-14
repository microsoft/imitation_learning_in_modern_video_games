"""Re-download CS:GO deathmatch data from the PUBLIC HF dataset
TeaPearce/CounterStrike_Deathmatch (no HF_TOKEN needed).

Files are packaged as 200-file tar chunks named
`hdf5_dm_july2021_<start>_to_<end>.tar` (~26 GB each). This extracts the .hdf5
(flattened) into --dest, skipping any file already present.

Usage:
  # first 2 chunks (files 1..400):
  python pixelbc/scripts/download_csgo_data.py --num-tars 2
  # specific chunks by name:
  python pixelbc/scripts/download_csgo_data.py --tars hdf5_dm_july2021_1_to_200.tar
  # list what's available without downloading:
  python pixelbc/scripts/download_csgo_data.py --list
"""
import argparse
import os
import re
import tarfile

from huggingface_hub import HfApi, hf_hub_download

REPO = "TeaPearce/CounterStrike_Deathmatch"
CHUNK_RE = re.compile(r"hdf5_dm_july2021_\d+_to_\d+\.tar$")


def start_num(name):
    m = re.search(r"_(\d+)_to_\d+\.tar$", name)
    return int(m.group(1)) if m else 10**9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-tars", type=int, default=1,
                   help="Download the first N 200-file chunks (ordered by file number).")
    p.add_argument("--tars", nargs="+", default=None,
                   help="Explicit tar filenames to download (overrides --num-tars).")
    p.add_argument("--dest", default=r"C:\csgo_data", help="Where the .hdf5 land (data_path).")
    p.add_argument("--stage", default=r"C:\csgo_dl", help="Where the .tar are downloaded to.")
    p.add_argument("--list", action="store_true", help="List available chunks and exit.")
    args = p.parse_args()

    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")
    chunks = sorted([f for f in all_files if CHUNK_RE.match(f)], key=start_num)

    print(f"{len(chunks)} deathmatch chunks available (200 files each), from {REPO}:")
    if args.list:
        for c in chunks:
            print("  ", c)
        return 0

    if args.tars:
        pick = [t for t in args.tars if t in chunks]
        missing = set(args.tars) - set(pick)
        if missing:
            print("  (requested but not found, skipped):", sorted(missing))
    else:
        pick = chunks[: args.num_tars]
    print(f"Downloading {len(pick)} chunk(s) (~26 GB each): {pick}")

    os.makedirs(args.dest, exist_ok=True)
    total_new = 0
    for t in pick:
        print(f"\n[download] {t}")
        local = hf_hub_download(REPO, t, repo_type="dataset", local_dir=args.stage)
        print(f"[extract ] {t} -> {args.dest}")
        with tarfile.open(local) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".hdf5"):
                    continue
                base = os.path.basename(m.name)
                if os.path.exists(os.path.join(args.dest, base)):
                    continue  # never overwrite existing
                m.name = base  # flatten out of any subdir
                tf.extract(m, args.dest)
                total_new += 1

    print(f"\nDone. {total_new} new .hdf5 extracted into {args.dest}.")
    print("Next: add the new filenames to pixelbc/data/paper_data_split/csgo_dm_train_files.txt,")
    print("then re-run precompute_dino_embeddings.bat + precompute_dino_depth_embeddings.bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
