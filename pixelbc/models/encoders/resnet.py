# modelled after https://arxiv.org/abs/2201.03545
# largely from humanmodelling/models/nn/model_blocks.py
import numpy as np
from torch import nn


class ConvNextBlock(nn.Module):
    """Conv layer which keeps the dimensionality the same."""

    def __init__(self, channels, activations="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=7 // 2, groups=channels)  # 'Depthwise' conv
        self.group_norm = nn.GroupNorm(channels, channels)  # Should be equivalent to layernorm

        # Transformer-style non-linearity
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, padding=0)
        if activations == "relu":
            self.activation = nn.ReLU()
        elif activations == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activations}")

        self.conv3 = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.group_norm(y)
        y = self.conv2(y)
        y = self.activation(y)
        y = self.conv3(y)
        return x + y


class ConvNextDownsample(nn.Module):
    """Conv layer which downsamples the image by a factor of 2.""" ""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.group_norm = nn.GroupNorm(c_in, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv1(self.group_norm(x))


class CustomResNet(nn.Module):
    def __init__(self, image_shape, enc_dim=2, channel_start=8):
        """
        Returns a ResNet model for image processing of squared images.
        :param image_shape: shape of the image as (channels, height, width)
        :param enc_dim: dimension of output encoding
        :param channel_start: number of channels in first layer
        """
        super().__init__()
        channels, image_height, image_width = image_shape
        assert image_width == image_height, "CNN encoder only supports squared images"
        self.img_dim = image_width
        self.enc_dim = enc_dim
        # check if img_dim is power of 2
        assert self.img_dim & (self.img_dim - 1) == 0, "ImageEncoder only supports images with dimensions that are a power of 2"
        assert enc_dim & (enc_dim - 1) == 0, "ImageEncoder only supports encoding dimensions that are a power of 2"
        assert enc_dim >= 2, "Image encoding dimension must be at least 2"

        # first layer + GeLU
        # input: (channels, img_dim, img_dim)
        # output: (channel_start, img_dim / 4, img_dim / 4)
        self.conv1 = nn.Conv2d(channels, channel_start, kernel_size=8, stride=4, padding=3)

        # number of downsampling layers until 2x2
        self.num_layers = int(np.log2(self.img_dim // 4)) - int(np.log2(enc_dim))
        x = channel_start
        for i in range(self.num_layers):
            self.add_module(f"block{i+1}", ConvNextBlock(x, activations="gelu"))
            self.add_module(f"downsample{i+1}", ConvNextDownsample(x, x * 2))
            x = x * 2

        # channel_start*2**num_layers x enc_dim x enc_dim
        self.output_dim = channel_start * 2**self.num_layers * enc_dim * enc_dim
        self.final_ln = nn.LayerNorm(self.output_dim)

    def get_embedding_dim(self):
        return self.output_dim

    def forward(self, x):
        assert len(x.shape) == 4, f"CustomResNet expects images of shape (batch, channels, {self.img_dim}, {self.img_dim})"
        img_channel, img_height, img_width = x.shape[-3:]
        assert self.img_dim == img_width == img_height, f"CustomResNet expects images of size (channels, {self.img_dim}, {self.img_dim})"
        x = nn.functional.gelu(self.conv1(x))
        for i in range(self.num_layers):
            x = getattr(self, f"block{i+1}")(x)
            x = getattr(self, f"downsample{i+1}")(x)
        flat_x = x.reshape(-1, self.output_dim)
        return self.final_ln(flat_x)
