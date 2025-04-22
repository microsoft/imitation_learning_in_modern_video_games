from lightning.pytorch.loggers import TensorBoardLogger

# Minimal set of keys to log if logging only minimal metrics is turned on.
MINIMAL_KEYS = [
    "epoch",
    "train/loss",
    "train/button_all_balanced_accuracy",
    "train/button_loss",
    "train/trigger_balanced_accuracy",
    "train/trigger_loss",
    "train/joystick_r2",
    "train/joystick_lx_r2",
    "train/joystick_ly_r2",
    "train/joystick_rx_r2",
    "train/joystick_ry_r2",
    "train/joystick_loss",
    "train/grad_norm",
    "val/loss",
    "val/button_all_balanced_accuracy",
    "val/button_loss",
    "val/trigger_balanced_accuracy",
    "val/trigger_loss",
    "val/joystick_r2",
    "val/joystick_lx_r2",
    "val/joystick_ly_r2",
    "val/joystick_rx_r2",
    "val/joystick_ry_r2",
    "val/joystick_loss",
]


def get_loggers(config, log_dir):
    """
    Define loggers for PyTorch Lightning during training.
    :param config: OmegaConf object containing the configuration for training
    :param log_dir: directory where the logs are saved
    :return: list of loggers
    """
    if config.log.minimal_metrics:
        loggers = [MinimalTensorBoardLogger(minimal_keys=MINIMAL_KEYS, save_dir=log_dir, name="lightning_logs")]
    else:
        loggers = [TensorBoardLogger(save_dir=log_dir, name="lightning_logs")]

    return loggers


class MinimalTensorBoardLogger(TensorBoardLogger):
    """
    A TensorBoardLogger that logs only a subset of metrics.
    """

    def __init__(self, minimal_keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimal_keys = minimal_keys

    def log_metrics(self, metrics, step=None):
        # Override the log_metrics method to prevent an excessive number of metrics from being logged.
        metrics_to_log = {k: v for k, v in metrics.items() if k in self.minimal_keys}
        super().log_metrics(metrics_to_log, step)
