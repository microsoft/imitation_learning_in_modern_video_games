import psutil
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only


class DeviceStatsMemoryMonitor(DeviceStatsMonitor):
    def _log_memory(self, model, log_name, log_sync_dist=False):
        memory_data = psutil.virtual_memory()
        model.log(
            f"MemoryProfiling.on_{log_name}_batch_end/memory_used_gb",
            memory_data.used / (1024 * 1024 * 1024),
            sync_dist=log_sync_dist,
        )
        model.log(
            f"MemoryProfiling.on_{log_name}_batch_end/memory_available_gb",
            memory_data.available / (1024 * 1024 * 1024),
            sync_dist=log_sync_dist,
        )
        model.log(f"MemoryProfiling.on_{log_name}_batch_end/memory_used_percent", memory_data.percent, sync_dist=log_sync_dist)
        memory_info = psutil.Process().memory_info()
        model.log(
            f"MemoryProfiling.on_{log_name}_batch_end/memory_used_gb_rss",
            memory_info.rss / (1024 * 1024 * 1024),
            sync_dist=log_sync_dist,
        )
        model.log(
            f"MemoryProfiling.on_{log_name}_batch_end/memory_used_gb_vms",
            memory_info.vms / (1024 * 1024 * 1024),
            sync_dist=log_sync_dist,
        )

    def on_train_batch_end(self, trainer, model, *args, **kwargs):
        super().on_train_batch_end(trainer, model, *args, **kwargs)
        self._log_memory(model, "train", False)

    def on_validation_batch_end(self, trainer, model, *args, **kwargs):
        super().on_validation_batch_end(trainer, model, *args, **kwargs)
        self._log_memory(model, "validation", True)

    def on_test_batch_end(self, trainer, model, *args, **kwargs):
        super().on_test_batch_end(trainer, model, *args, **kwargs)
        self._log_memory(model, "test", True)


class WeightsCallback(Callback):
    """
    Logs the size of the weights/biases.
    """

    @rank_zero_only
    def __init__(self, log_freq, log_scalars=False):
        self.log_freq = log_freq
        self.timestep = -1
        self.log_scalars = log_scalars

    @rank_zero_only
    def on_after_backward(self, trainer, model):
        self.timestep += 1
        # This logging is a little heavy, so we don't want to do it all the time
        if self.timestep % self.log_freq != 0:
            return
        tb_logger = trainer.logger  # This seems to always return the first logger, which is the tensorboard logger for us.
        if not isinstance(tb_logger, TensorBoardLogger):
            print("The first logger is not a TensorBoardLogger. Skipping weights logging.")
            return
        for name, param_model in model.named_parameters():
            param = param_model.detach()
            tb_logger.experiment.add_histogram(f"model/{name}", param.data.cpu().numpy(), trainer.global_step)
            if self.log_scalars:
                model.log(f"parameters/model_l2/{name}", param.data.norm(2).item(), on_step=True, on_epoch=False)
                model.log(f"parameters/model_min/{name}", param.data.min().item(), on_step=True, on_epoch=False)
                model.log(f"parameters/model_max/{name}", param.data.max().item(), on_step=True, on_epoch=False)


def get_callbacks(log_config, log_dir, checkpoint_dir):
    """
    Define callbacks for checkpoints during training.
    :param log_config: OmegaConf object containing the config for logging
    :param log_dir: directory where the logs are saved
    :param checkpoint_dir: directory where the checkpoints are saved
    :return: list of callbacks
    """
    callbacks = []

    if log_config.checkpoint_top_k_validation > 0:
        # saves top-K checkpoints based on "val_loss" metric
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=log_config.checkpoint_top_k_validation,
                monitor="val/loss",
                mode="min",
                filename="checkpoint-epoch={epoch:02d}-val_loss={val/loss:.2f}",
                auto_insert_metric_name=False,
            )
        )
    # saves checkpoints every N epochs,
    if log_config.checkpoint_epoch_freq > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="step",
                save_top_k=log_config.checkpoint_epoch_freq,
                every_n_epochs=log_config.checkpoint_epoch_freq,
                filename="checkpoint-{epoch:02d}",
            )
        )
    if log_config.checkpoint_last:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, filename="last", every_n_epochs=1))

    if log_config.weights:
        callbacks.append(WeightsCallback(log_config.weights_step_freq, log_config.weights_log_scalars))

    if log_config.device_stats_monitor:
        callbacks.append(DeviceStatsMemoryMonitor())

    return callbacks
