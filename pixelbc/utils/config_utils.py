import shutil
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pixelbc.data.data_split import get_file_paths_from_split_file


def _load_default_config(config_path):
    # load default config relative to given config
    default_config_path = Path(config_path).parent / "default.yaml"
    assert default_config_path.exists(), f"Default config was expected at {default_config_path} but not found!"
    return OmegaConf.load(default_config_path)


def load_config_from_file(config_path):
    """
    Load config from file and merge with default config
    :param config_path: Path to the config.
    :return: The configuration as OmegaConf config object
    """
    # load config from file
    config_path = Path(config_path)
    assert config_path.exists(), f"Config file {config_path} does not exist"
    config = OmegaConf.load(config_path)
    # merge configs with priority of default config < given config
    default_config = _load_default_config(config_path)
    config = OmegaConf.merge(default_config, config)
    return config


def load_config_from_file_and_cli(argv):
    config = load_config_from_file(argv[1])

    # parse command line arguments
    cli_config = OmegaConf.from_cli()

    # merge configs with priority of default config < given config < command line arguments
    config = OmegaConf.merge(config, cli_config)
    _check_and_update_config(config)

    # Add config path to config
    config.config_path = argv[1]

    return config


def _check_and_update_config(config):
    if "image_shape" not in config.data:
        assert (
            "image_width" in config.data and "image_height" in config.data and "framestacking" in config.data
        ), "Data config missing `image_shape` or `image_width` and `image_height` and `framestacking`"
        config.data.image_shape = (3 * config.data.framestacking, config.data.image_height, config.data.image_width)
    data_config = config.data

    # set mixed precision in lightning trainer config if available
    if "precision" in config.trainer:
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, setting precision to 32")
            config.trainer.precision = 32
        else:
            print(f"Setting precision to {config.trainer.precision}")

    # add metadata of data to model config
    for key, value in data_config.items():
        config.model[key] = value


def has_train_data(config):
    if config.data.train_split_file_path is None:
        return False
    else:
        return get_file_paths_from_split_file(config.data.train_split_file_path, config.data.data_path)


def has_validation_data(config):
    if config.data.validation_split_file_path is None:
        return False
    else:
        return get_file_paths_from_split_file(config.data.validation_split_file_path, config.data.data_path)


def has_test_data(config):
    if config.data.test_split_file_path is None:
        return False
    else:
        return get_file_paths_from_split_file(config.data.test_split_file_path, config.data.data_path)


def save_full_train_config_and_split(config, trainer):
    with open(Path(trainer.log_dir) / "train_config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # save data split to log directory
    data_root_dir = config.data
    if data_root_dir.train_split_file_path is not None:
        shutil.copy(
            Path(__file__).parent.parent / Path("data") / data_root_dir.train_split_file_path,
            Path(trainer.log_dir) / "train_split.txt",
        )
    if data_root_dir.validation_split_file_path is not None:
        shutil.copy(
            Path(__file__).parent.parent / "data" / data_root_dir.validation_split_file_path,
            Path(trainer.log_dir) / "validation_split.txt",
        )
    if data_root_dir.test_split_file_path is not None:
        shutil.copy(
            Path(__file__).parent.parent / "data" / data_root_dir.test_split_file_path,
            Path(trainer.log_dir) / "test_split.txt",
        )
