import torch
from omegaconf import OmegaConf

from pixelbc.models.bc_model import BCModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_backwards_compatibility_kwargs(hyperparameters):
    """
    From checkpoint hyperparameters dictionary, extract all hyperparameters that are needed for backwards compatibility
    :param hyperparameters: dictionary of hyperparameters
    :return: dictionary of hyperparameters that are needed for backwards compatibility
    """
    kwargs = {}
    if "image_shape" not in hyperparameters:
        # old config does not contain image_shape
        assert (
            "framestacking" in hyperparameters and "image_width" in hyperparameters and "image_height" in hyperparameters
        ), "Need to provide `img_shape` as hyperparameter or 'framestacking', 'image_width', and 'image_height'!"
        image_shape = (
            hyperparameters["framestacking"] * 3,
            hyperparameters["image_height"],
            hyperparameters["image_width"],
        )
        kwargs["image_shape"] = image_shape

    if "encoder" not in hyperparameters and "pretrained_encoder" not in hyperparameters:
        # config from before encoder/ pretrained encoder config refactoring
        assert "image_encoder" in hyperparameters, "Old config does not contain `image_encoder`!"
        assert (
            "cnn_encoder_dim" in hyperparameters and "cnn_encoder_start_channels" in hyperparameters
        ), "Old config does not contain `cnn_encoder_dim` or 'cnn_encoder_start_channels'!"
        assert (
            "mlp_encoder_hidden_size" in hyperparameters and "mlp_encoder_num_layers" in hyperparameters
        ), "Old config does not contain `mlp_encoder_hidden_size` or 'mlp_encoder_num_layers'!"
        encoder_config = OmegaConf.create(
            {
                "type": hyperparameters["image_encoder"],
                "cnn_encoder_dim": hyperparameters["cnn_encoder_dim"],
                "cnn_encoder_start_channels": hyperparameters["cnn_encoder_start_channels"],
                "mlp_encoder_hidden_size": hyperparameters["mlp_encoder_hidden_size"],
                "mlp_encoder_num_layers": hyperparameters["mlp_encoder_num_layers"],
                "use_image_augmentation": hyperparameters["use_image_augmentation"] if "use_image_augmentation" in hyperparameters else False,
            }
        )
        kwargs["encoder"] = encoder_config
        kwargs["pretrained_encoder"] = None

    if "bc_policy" not in hyperparameters:
        # config from before config refactoring
        assert (
            "bc_hidden_size" in hyperparameters and "bc_num_layers" in hyperparameters
        ), "Old config does not contain `bc_hidden_size` or 'bc_num_layers'!"
        bc_policy_config = OmegaConf.create(
            {
                "type": "mlp",
                "hidden_size": hyperparameters["bc_hidden_size"],
                "num_layers": hyperparameters["bc_num_layers"],
            }
        )
        kwargs["bc_policy"] = bc_policy_config

    kwargs["train_from_embeddings"] = False
    if "sequence_length" not in hyperparameters:
        kwargs["sequence_length"] = 1
    if "lr_warmup_steps" not in hyperparameters:
        kwargs["lr_warmup_steps"] = 0

    return kwargs


def load_checkpoint(checkpoint_path, eval_mode=False, device=DEVICE):
    """
    Load a checkpoint from a file.
    :param checkpoint_path: path to the checkpoint file
    :return: model, hyperparameters
    """
    try:
        hyperparameters = torch.load(checkpoint_path, map_location=device)["hyper_parameters"]
        if eval_mode:
            if "encoder" in hyperparameters:
                encoder_config = hyperparameters["encoder"]
                # disable image augmentation for evaluation
                if encoder_config is not None:
                    encoder_config = OmegaConf.create(encoder_config)
                    encoder_config.use_image_augmentation = False
                model = BCModel.load_from_checkpoint(checkpoint_path, encoder=encoder_config, train_from_embeddings=False, map_location=device)
            else:
                model = BCModel.load_from_checkpoint(checkpoint_path, train_from_embeddings=False, map_location=device)
            model.eval()
        else:
            model = BCModel.load_from_checkpoint(checkpoint_path, map_location=device)
    except TypeError as e:
        print(f"Error loading model from checkpoint {checkpoint_path}: {e}")
        print("Trying to modify checkpoint config to match current config...")
        hyperparameters = torch.load(checkpoint_path, map_location=device)["hyper_parameters"]
        compatibility_kwargs = get_backwards_compatibility_kwargs(hyperparameters)
        hyperparameters.update(compatibility_kwargs)
        model = BCModel.load_from_checkpoint(checkpoint_path, **compatibility_kwargs, map_location=device)
    hyperparameters = OmegaConf.create(hyperparameters)
    return model, hyperparameters
