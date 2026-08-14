#!/usr/bin/env python
"""Run held-out validation on a trained checkpoint (no training).

Loads the config + checkpoint, builds the datamodule on the validation split,
runs trainer.validate(), and prints the val/* metrics.

Usage:
  python pixelbc/validate.py pixelbc/configs/csgo/default.yaml \
      checkpoint=training_logs/default_seed_42/models/last-v3.ckpt \
      data.data_path=C:\csgo_data trainer.accelerator=gpu trainer.devices=1
"""
import sys

import lightning.pytorch as pl

from pixelbc.data import get_data_module
from pixelbc.models.bc_model import BCModel
from pixelbc.utils.config_utils import has_validation_data, load_config_from_file_and_cli


def main(argv):
    config = load_config_from_file_and_cli(argv)
    pl.seed_everything(config.seed, workers=True)

    assert has_validation_data(config), (
        "No validation data found -- set data.validation_split_file_path to a split "
        "file (and for RGBD, run make_depth.py on those files first)."
    )
    ckpt = config.checkpoint if "checkpoint" in config and config.checkpoint is not None else None
    assert ckpt is not None, "Pass the checkpoint to score, e.g. checkpoint=path/to/model.ckpt"

    dm = get_data_module(config.data, config.model)
    model = BCModel(**config.model)

    trainer = pl.Trainer(
        accelerator=config.trainer.get("accelerator", "auto"),
        devices=config.trainer.get("devices", 1),
        precision=config.trainer.get("precision", "16-mixed"),
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
    )
    print(f"Validating checkpoint: {ckpt}")
    results = trainer.validate(model=model, datamodule=dm, ckpt_path=ckpt)

    print("\n==== validation metrics ====")
    for d in results:
        for k in sorted(d):
            print(f"  {k:42s} {float(d[k]):.5f}")


if __name__ == "__main__":
    main(sys.argv)
