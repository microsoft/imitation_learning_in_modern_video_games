import warnings
from abc import ABC, abstractmethod

import torch
from torch import nn

from pixelbc.models.encoders.dqn_cnn import NatureCNN
from pixelbc.models.encoders.impala_resnet import ImpalaResNet
from pixelbc.models.encoders.resnet import CustomResNet
from pixelbc.models.encoders.vit import ViT
from pixelbc.models.utils.image_augmentations import default_transform
from pixelbc.models.utils.model_utils import MLP


class ImageEncoderWrapper(nn.Module):
    def __init__(self, encoder, process):
        super().__init__()
        self.encoder = encoder
        self.process = process

    def get_embedding_dim(self):
        return self.encoder.get_embedding_dim()

    def forward(self, x):
        # resize to (batch_size, channels, height, width) for processing and encoder
        assert len(x.shape) >= 3, "ImageEncoderWrapper expects at least 3 dimensions (..., channels, height, width)"
        batch_shape = x.shape[:-3]
        img_channel, img_height, img_width = x.shape[-3:]
        x = x.reshape(-1, img_channel, img_height, img_width)

        # process image and ensure types
        x = self.process(x)
        assert torch.is_tensor(x) and x.dtype == torch.float32, "ImageEncoderWrapper expects float32 tensor"
        if x.sum() == 0:
            warnings.warn("Image augmentation process returned all 0s! If the screen was not fully black, this is likely a bug!")

        # embed image and reshape to original dimensions
        emb = self.encoder(x)
        return emb.reshape(*batch_shape, -1)


class ImageEncoder(nn.Module, ABC):
    """Abstract class for image encoders."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_embedding_dim():
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        pass


class MLPImageEncoder(ImageEncoder):
    def __init__(self, image_shape, hidden_size, num_layers):
        super().__init__()
        channels, image_height, image_width = image_shape
        self.input_dim = channels * image_height * image_width
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = MLP(self.input_dim, self.hidden_size, self.num_layers)

    def get_embedding_dim(self):
        return self.hidden_size

    def forward(self, x):
        return self.model(x)


class ResNetImageEncoder(ImageEncoder):
    def __init__(self, image_shape, encoding_dim=2, start_channels=8):
        super().__init__()
        self.model = CustomResNet(image_shape, encoding_dim, start_channels)
        self.embedding_dim = self.model.output_dim

    def get_embedding_dim(self):
        return self.embedding_dim

    def forward(self, x):
        return self.model(x)


class ViTImageEncoder(ImageEncoder):
    def __init__(
        self,
        image_shape,
        patch_size,
        dim,
        num_layers,
        num_heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        channels, image_height, image_width = image_shape
        assert image_height == image_width, "ViT expects square images"
        self.image_channels = channels
        self.image_size = image_height
        self.emb_dim = dim

        self.model = ViT(
            image_size=self.image_size,
            patch_size=patch_size,
            dim=dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

    def get_embedding_dim(self):
        return self.emb_dim

    def forward(self, x):
        return self.model(x)


def get_image_encoder(image_shape, encoder_config):
    """
    Returns an encoder for the image input.
    :param image_shape: shape of the image as (channels, height, width)
    :param encoder_config: config for the encoder
    :return: image encoder
    """
    assert "type" in encoder_config, "Encoder config must have a type"
    if encoder_config.type == "nature_cnn":
        encoder = NatureCNN(image_shape)
    elif encoder_config.type == "impala_resnet":
        encoder = ImpalaResNet(image_shape)
    elif encoder_config.type == "cnn" or encoder_config.type == "resnet":
        encoder = ResNetImageEncoder(image_shape, encoder_config.cnn_encoder_dim, encoder_config.cnn_encoder_start_channels)
    elif encoder_config.type == "vit":
        encoder = ViTImageEncoder(
            image_shape,
            encoder_config.vit_encoder_patch_size,
            encoder_config.vit_encoder_dim,
            encoder_config.vit_encoder_num_layers,
            encoder_config.vit_encoder_num_heads,
            encoder_config.vit_encoder_mlp_dim,
        )
    elif encoder_config.type == "mlp":
        encoder = MLPImageEncoder(image_shape, encoder_config.mlp_encoder_hidden_size, encoder_config.mlp_encoder_num_layers)
    else:
        raise ValueError(f"Unknown image encoder {encoder_config.type}")
    process = default_transform(encoder_config.use_image_augmentation)
    return ImageEncoderWrapper(encoder, process)
