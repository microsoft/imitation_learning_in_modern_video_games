#!/usr/bin/env python3
"""Stream a CS:GO deathmatch tar over plain HTTP and extract .hdf5 as it arrives.

Bypasses HF's Xet backend (which can wedge when two Xet downloads share the chunk
cache) by doing a direct `resolve/main/<tar>` GET and unpacking the tar as a
stream. Because a tar stores files sequentially (~130 MB each), --max-files N
stops after N files -> you download ~N*130 MB instead of the full ~26 GB.

Skips any .hdf5 already present in --dest. The repo is public, so HF_TOKEN is
optional (used if set).

Run:
  python download_csgo_stream.py --tar hdf5_dm_july2021_401_to_600.tar --dest C:/csgo_data_new --max-files 50
"""
import argparse
import os
import sys
import tarfile
import time

import requests

BASE = "https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main"

# Full deathmatch .hdf5 are ~130 MB; anything smaller on disk is a truncated
# partial from a killed run and must be re-downloaded, not skipped.
_MIN_COMPLETE_BYTES = 100 * 1024 * 1024


def _stream_one_tar(tar, dest, headers, budget, t0, extracted):
    """Stream one tar, extracting new .hdf5 until the GLOBAL budget is hit.
    Returns (updated extracted count, hit_budget bool)."""
    url = f"{BASE}/{tar}"
    print(f"\nstreaming {url}", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True  # transparently handle any content-encoding
        # mode 'r|' = streaming, sequential read (no seeking) -> extract as it flows.
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for member in tf:
                if not member.name.endswith(".hdf5"):
                    continue
                base = os.path.basename(member.name)
                out_path = os.path.join(dest, base)
                # Skip only if a PLAUSIBLY-COMPLETE file already exists; a small
                # file is a truncated partial from a killed run -> re-download it.
                if os.path.exists(out_path) and os.path.getsize(out_path) >= _MIN_COMPLETE_BYTES:
                    print(f"  skip (exists) {base}", flush=True)
                    continue
                fsrc = tf.extractfile(member)
                if fsrc is None:
                    continue
                # Write to a .part temp, then atomically rename over the final
                # name -- a Ctrl-C leaves a .part (ignored next run), never a
                # corrupt final .hdf5.
                tmp_path = out_path + ".part"
                with open(tmp_path, "wb") as fout:
                    while True:
                        buf = fsrc.read(1024 * 1024)
                        if not buf:
                            break
                        fout.write(buf)
                os.replace(tmp_path, out_path)
                extracted += 1
                mb = member.size / 1e6
                rate = extracted / max(1e-6, time.time() - t0)
                print(f"  [{extracted:4}] {base}  ({mb:.0f} MB)  {rate:.2f} files/s", flush=True)
                if budget > 0 and extracted >= budget:
                    print(f"\nreached --max-files {budget}; stopping.", flush=True)
                    return extracted, True
    return extracted, False


# Zero-overlap gap chunks for the current 329-file dataset, in order. Spanning
# these gives up to 600 brand-new files; --max-files caps the global total.
DEFAULT_TARS = [
    "hdf5_dm_july2021_401_to_600.tar",
    "hdf5_dm_july2021_601_to_800.tar",
    "hdf5_dm_july2021_801_to_1000.tar",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Stream-extract CS:GO tars (Xet-free)")
    p.add_argument("--tars", nargs="+", default=DEFAULT_TARS,
                   help="tar filenames to stream in order (200 files each)")
    p.add_argument("--dest", default=r"C:/csgo_data_new", help="where .hdf5 land")
    p.add_argument("--max-files", type=int, default=500,
                   help="GLOBAL cap across all tars (default 500; -1 = every file)")
    a = p.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"tars={a.tars}\n  -> {a.dest}  (max_files={a.max_files})")
    t0 = time.time()
    extracted = 0
    for tar in a.tars:
        extracted, hit = _stream_one_tar(tar, a.dest, headers, a.max_files, t0, extracted)
        if hit:
            break

    dt = time.time() - t0
    print(f"\nDone. Extracted {extracted} new .hdf5 into {a.dest} in {dt / 60:.1f} min.")
    print("Next: move into C:\\csgo_data, add names to csgo_dm_train_files.txt, precompute embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
