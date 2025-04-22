import datetime
import os
import sys

import lightning.pytorch as pl
import torch

from pixelbc.data import get_data_module
from pixelbc.models.bc_model import BCModel
from pixelbc.utils.callbacks import get_callbacks
from pixelbc.utils.config_utils import (has_test_data, has_train_data,
                                        has_validation_data,
                                        load_config_from_file_and_cli,
                                        save_full_train_config_and_split)
from pixelbc.utils.loggers import get_loggers

DEFAULT_LOG_DIR = "training_logs"
DEFAULT_CHECKPOINT_DIR = "models"


def get_log_and_checkpoint_dir(config):
    log_dir = config.log_dir if "log_dir" in config else DEFAULT_LOG_DIR
    checkpoint_dir = config.checkpoint_dir if "checkpoint_dir" in config else DEFAULT_CHECKPOINT_DIR
    config_path = config.config_path
    # Filename without extension
    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    run_name = config.run_name if "run_name" in config else config_filename
    seed = config.seed

    log_dir = f"{log_dir}/{run_name}_seed_{seed}"
    checkpoint_dir = f"{log_dir}/{DEFAULT_CHECKPOINT_DIR}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Local run -- Logging data to {log_dir} and checkpoints to {checkpoint_dir}")

    return log_dir, checkpoint_dir


def main(argv):
    if len(argv) < 2:
        print("Usage: run_train.py <path/to/config.yaml>")
        sys.exit(2)

    config = load_config_from_file_and_cli(argv)
    pl.seed_everything(config.seed, workers=True)

    # inititate data module and IL model
    dm = get_data_module(config.data, config.model)
    model = BCModel(**config.model)
    if "checkpoint" in config and config.checkpoint is not None:
        print(f"Loading model from checkpoint {config.checkpoint}")
        model = model.load_from_checkpoint(config.checkpoint)
    print(model)

    # Documentation: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # "highest" would be original float32 setting
    # "high" seems to use faster algorithms if available
    torch.set_float32_matmul_precision("high")

    # create callbacks, trainer and start training
    assert has_train_data(config), "No training data found according to config."
    log_dir, checkpoint_dir = get_log_and_checkpoint_dir(config)
    callbacks = get_callbacks(config.log, log_dir=log_dir, checkpoint_dir=checkpoint_dir)
    loggers = get_loggers(config, log_dir=log_dir)
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
        limit_val_batches=0.0 if not has_validation_data(config) else None,
        strategy=pl.strategies.DDPStrategy(static_graph=False, timeout=datetime.timedelta(seconds=3600)),
    )
    trainer.fit(model=model, datamodule=dm, **config.fit if "fit" in config and config.fit is not None else {})
    # test model if test data is provided
    if has_test_data(config):
        trainer.test(model=model, datamodule=dm)

    # save full config to log directory
    save_full_train_config_and_split(config, trainer)


if __name__ == "__main__":
    main(sys.argv)
