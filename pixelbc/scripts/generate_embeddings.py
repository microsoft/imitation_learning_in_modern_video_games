#!/usr/bin/env python3
"""This is the script for generating image embeddings using a pretrained encoder.

We load images from all video files in a sequential order and save embeddings in a specified folder

Example:
    >>> CUDA_VISIBLE_DEVICES=0 \
    >>> scripts/generate_embeddings.py \
    >>> --game csgo \
    >>> --encoder_config configs/csgo/csgo_dino_test.yaml \
    >>> --filelist data/csgo_test/train.txt data/csgo_test/validation.txt data/csgo_test/test.txt \
    >>> --data_base_path /path/to/data \
    >>> --output_dir /path/to/data \  # will create /path/to/data/embeddings folder under which to store embeddings
    >>> --batch_size 4096  # set to the maximum number of steps that can be processed in one batch without OOM

Tips:
If things are too slow on a single GPU, you can try the following options:
1. Set MAX_PROCESS_STEPS to a larger number (e.g. 8192)
2. Utilize multiple processes with different `filelists` and CUDA_VISIBLE_DEVICES.

"""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pixelbc.data.data_parsers import get_data_parser
from pixelbc.data.data_split import get_file_paths_from_split_file
from pixelbc.data.utils import get_pretrained_encoder_dirname
from pixelbc.models.encoders.pretrained_encoders import get_pretrained_encoder
from pixelbc.utils.config_utils import load_config_from_file

# Key for saving preprocessed data in the config file to avoid name conflicts
CONFIG_CUSTOM_DATA_KEY = "custom_data"
# Maximum number of steps to process in one batch (to avoid OOM)
# 8192: tested on an A6000 with 48GB VRAM to process 128x128 pixel images to 384-dim embeddings
# This can be larger if you have more VRAM :)
MAX_PROCESS_STEPS = 2048


def _parse_args():
    parser = argparse.ArgumentParser(description="Embed trajectories")
    # Dataset size and split
    parser.add_argument(
        "--game",
        type=str,
        default="minerl",
        choices=["minerl", "csgo"],
        help="Game data is for.",
    )
    parser.add_argument(
        "--encoder_config",
        type=str,
        required=True,
        help="Path to the config YAML file containing encoder settings. The format of this config should be the same as the one used to train models.",
    )
    parser.add_argument(
        "--encoder_family",
        type=str,
        default=None,
        help="Overwrite encoder family of config to generate embeddings for.",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=None,
        help="Overwrite encoder name of config to generate embeddings for.",
    )
    parser.add_argument(
        "--data_base_path",
        type=str,
        default=None,
        help="Base path to add to files in the file list.",
    )
    parser.add_argument(
        "--filelist",
        nargs="+",
        help="Paths to the files containing the list of files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the embeddings will be saved. If the directory does not exist, it will be created.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=-1,
        help="How many of the trajectories from filelists to use (starting from the top of the file). If -1, all files will be used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=MAX_PROCESS_STEPS,
        help="Batch size for processing the data. This should be set to the maximum number of steps that can be processed in one batch without OOM.",
    )
    return parser.parse_args()


def _sanity_check_and_prepare_resources(args):
    # sanity check
    # valid data paths

    config = load_config_from_file(args.encoder_config)
    assert "data" in config, f"Config file {args.encoder_config} does not contain a 'data' section!"
    assert "model" in config, f"Config file {args.encoder_config} does not contain a 'model' section!"
    assert "pretrained_encoder" in config.model, f"Config file {args.encoder_config} does not contain a 'pretrained_encoder' section!"

    # overwrite encoder family and name if argument is given
    if args.encoder_family is not None:
        config.model.pretrained_encoder.family = args.encoder_family
    if args.encoder_name is not None:
        config.model.pretrained_encoder.name = args.encoder_name
    config.model.train_from_embeddings = False

    config.data.game = args.game

    encoder_dirname = get_pretrained_encoder_dirname(config.model.pretrained_encoder.family, config.model.pretrained_encoder.name)
    output_dir = Path(args.output_dir) / encoder_dirname

    # output directory should be empty or non-existent
    assert not (output_dir.exists() and len(list(output_dir.iterdir())) > 0), f"Output directory {output_dir} is not empty!"

    # prepare resources
    # create output_dir if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # add another field to the config for customized data
    config[CONFIG_CUSTOM_DATA_KEY] = {}
    # load files to process
    filenames = None
    if args.filelist is not None and len(args.filelist) > 0:
        filenames = []
        for file_path in args.filelist:
            assert Path(file_path).exists(), f"File {file_path} does not exist!"
            if args.data_base_path is not None:
                paths = get_file_paths_from_split_file(file_path, base_path=args.data_base_path, relative_path=False)
            else:
                paths = get_file_paths_from_split_file(file_path, relative_path=False)
            assert all(Path(file_path).exists() for file_path in paths), f"File paths in {file_path} do not exist!"
            filenames.extend(paths)
        # truncate if necessary
        if args.num_files > 0:
            filenames = filenames[: args.num_files]

    config[CONFIG_CUSTOM_DATA_KEY]["file_list"] = filenames

    return config, output_dir


class SequentialDataLoader:
    def __init__(self, data_parser, file_list, batch_size=1):
        self.data_parser = data_parser
        self.file_list = file_list
        self.batch_size = batch_size

    def __iter__(self):
        for file in tqdm(self.file_list, desc="Embedding Trajectories", unit="trajs"):
            images, actions = self.data_parser.get_all_data(file)
            # add batch dimension
            images = images.unsqueeze(0)
            actions = actions.unsqueeze(0)
            yield (file, images, actions)

    def __len__(self):
        return len(self.file_list)


def setup_dataloader(config):
    assert not config.model.train_from_embeddings, "This script is for generating embeddings, set `model.train_from_embeddings=False`."
    file_list = config[CONFIG_CUSTOM_DATA_KEY]["file_list"]
    parser = get_data_parser(config.model, **config.data)
    return SequentialDataLoader(parser, file_list, batch_size=1)


def save_traj_embeddings(embeddings, output_dir, ref_filename):
    saved_file_path = Path(output_dir) / f"{Path(ref_filename).stem}.npz"
    parent_folder = saved_file_path.parent
    parent_folder.mkdir(parents=True, exist_ok=True)
    with open(saved_file_path, "wb") as f:
        np.savez_compressed(f, embedding=embeddings)


def split_trajectory(image_batch, split_length=MAX_PROCESS_STEPS):
    # Take a trajectory batch and split it into multiple subtrajectories if it is too long
    _, traj_length, _, _, _ = image_batch.shape
    if traj_length <= split_length:
        subtraj_list = [image_batch]
    else:
        # split the trajectory into multiple subtrajectories
        num_subtraj = ceil(traj_length / split_length)
        subtraj_list = []
        for i in range(num_subtraj):
            subtraj = image_batch[:, i * split_length : (i + 1) * split_length, :, :, :]
            subtraj_list.append(subtraj)
    return subtraj_list


def encode_trajectory(image_traj, encoder, device):
    # encode a trajectory and return the embedding as a numpy array
    with torch.no_grad():
        img_enc = encoder(image_traj.to(device))
        np_enc = img_enc.squeeze(dim=0).detach().cpu().numpy()
    return np_enc


def generate_embeddings(args, config, output_dir, device):
    dataloader = setup_dataloader(config)

    # load the pretrained encoder
    pretrained_encoder = get_pretrained_encoder(config.model.pretrained_encoder)
    pretrained_encoder.eval()
    pretrained_encoder = pretrained_encoder.to(device)

    for filename, batch_images, _ in dataloader:
        try:
            # split the trajectory into multiple subtrajectories if it is too long
            images_split_list = split_trajectory(batch_images, split_length=args.batch_size)
            # encode each subtrajectory
            embeddings_list = [encode_trajectory(images, pretrained_encoder, device) for images in images_split_list]
            # concatenate the embeddings of the subtrajectories
            embeddings = np.vstack(embeddings_list)
            save_traj_embeddings(embeddings, output_dir, filename)
        except torch.cuda.OutOfMemoryError:
            print(f"\tOut of memory when processing {filename}--> empty cache. Try setting --batch_size to a smaller value.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\tUnexpected exception[{e}] --> stop processing")
            raise


if __name__ == "__main__":
    args = _parse_args()
    config, output_dir = _sanity_check_and_prepare_resources(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        generate_embeddings(args, config, output_dir, device)
