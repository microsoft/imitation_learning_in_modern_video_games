"""Generate frozen-DINOv2 embeddings from the DEPTH channel of the CSGO data.

Mirrors generate_embeddings.py exactly, but feeds each frame's `frame_i_depth`
map (replicated 1->3 channels) through the same frozen encoder instead of RGB.
Output .npz have the same shape/format as the RGB embeddings, so the RGB+D
data loader can concat them per-frame.

Usage (see precompute_dino_depth_embeddings.bat):
  python pixelbc/scripts/generate_depth_embeddings.py \
      --encoder_config pixelbc/configs/csgo/aim_expert_dino_vitb.yaml \
      --data_base_path C:/csgo_data \
      --filelist <train> <val> <test> \
      --output_dir C:/csgo_data --encoder_subdir dino_depth
"""
import argparse
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from pixelbc.data.data_parsers import DataParser, get_data_parser
from pixelbc.models.encoders.pretrained_encoders import get_pretrained_encoder
from pixelbc.scripts.generate_embeddings import (encode_trajectory,
                                                 save_traj_embeddings,
                                                 split_trajectory)
from pixelbc.utils.config_utils import load_config_from_file

MAX_PROCESS_STEPS = 2048


def depth_get_all_data(self, action_filename):
    """Read frame_i_depth (HxW uint8), replicate to 3 channels, preprocess with
    the same fn the RGB path uses, and return (frames, actions)."""
    all_indices = range(self.NUM_STEPS_PER_FILE)
    frames, actions = [], []
    with h5py.File(action_filename, "r") as f:
        for idx in all_indices:
            depth_key = f"{self.KEY_INITIAL}{idx}_depth"
            if depth_key in f:
                depth = f[depth_key][:].astype(np.uint8)
            else:
                depth = np.zeros((150, 280), dtype=np.uint8)
            frame = np.repeat(depth[:, :, None], 3, axis=2)  # HxWx3
            frames.append(self.preprocess_fn(frame))
            actions.append(self._preprocess_action(f[self._get_action_key(idx)][:]))
    inputs = np.array(frames)
    actions = np.array(actions)
    return (
        torch.from_numpy(inputs).to(dtype=DataParser.IMAGE_TYPE),
        torch.from_numpy(actions).to(dtype=DataParser.ACTION_TYPE),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_config", required=True)
    p.add_argument("--data_base_path", required=True)
    p.add_argument("--filelist", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder_subdir", default="dino_depth",
                   help="Subdir under <output_dir>/embeddings/ to store depth embeddings.")
    p.add_argument("--batch_size", type=int, default=MAX_PROCESS_STEPS)
    args = p.parse_args()

    config = load_config_from_file(args.encoder_config)
    # image parser (not embeddings) with RGB (3-ch) preprocessing
    config.model.train_from_embeddings = False
    config.data.use_depth = False
    config.data.data_path = args.data_base_path

    parser = get_data_parser(config.model, **config.data)
    parser.get_all_data = types.MethodType(depth_get_all_data, parser)

    fam = config.model.pretrained_encoder.name.lower().replace("/", "")
    out_dir = Path(args.output_dir) / "embeddings" / args.encoder_subdir / fam
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Depth embeddings -> {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_pretrained_encoder(config.model.pretrained_encoder).eval().to(device)

    files = []
    for fl in args.filelist:
        with open(fl) as fh:
            files += [str(Path(args.data_base_path) / ln.strip()) for ln in fh if ln.strip()]
    files = sorted(set(files))
    print(f"{len(files)} files to embed (depth).")

    from tqdm import tqdm
    for fn in tqdm(files, desc="Depth-embedding", unit="trajs"):
        images, _ = parser.get_all_data(fn)
        images = images.unsqueeze(0)
        subtrajs = split_trajectory(images, split_length=args.batch_size)
        emb = np.vstack([encode_trajectory(s, encoder, device) for s in subtrajs])
        save_traj_embeddings(emb, out_dir, fn)
    print("Done.")


if __name__ == "__main__":
    main()
