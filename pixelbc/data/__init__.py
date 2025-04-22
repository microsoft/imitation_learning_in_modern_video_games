from pixelbc.data.datamodule import BCDataModule


def get_data_module(data_config, model_config):
    return BCDataModule(data_config, model_config)
