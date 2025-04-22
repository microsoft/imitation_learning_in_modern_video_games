# Source: https://github.com/Miffyli/minecraft-bc-2020/blob/master/torch_codes/modules.py

import math

import torch
import torch.nn.functional as F
from torch import nn

# References:
# [1] IMPALA. https://arxiv.org/pdf/1802.01561.pdf
# [2] R2D3. https://arxiv.org/pdf/1909.01387.pdf
# [3] Unixpickle's work https://github.com/amiranas/minerl_imitation_learning/blob/master/model.py#L104


class ResidualBlock(nn.Module):
    """
    Residual block from R2D3/IMPALA

    Taken from [1,2]
    """

    def __init__(self, num_channels, first_conv_weight_scale):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Copy paste from [3]
        self.bias1 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.scale = nn.Parameter(torch.ones([num_channels, 1, 1]))

        # Removed initialization from [3] to keep consistent with other models
        # # FixUp init (part of it):
        # #  - Final Convs in residual branches initialized
        # #    to zero
        # #  - Other convs in residual branches initialized
        # #    to a scaled value
        # #  - Biases handled manually as in [3]
        # with torch.no_grad():
        #     self.conv2.weight *= 0
        #     self.conv1.weight *= first_conv_weight_scale

    def forward(self, x):
        x = F.relu(x, inplace=True)
        original = x

        # Copy/paste from [3]
        x = x + self.bias1
        x = self.conv1(x)
        x = x + self.bias2

        x = F.relu(x, inplace=True)

        x = x + self.bias3
        x = self.conv2(x)
        x = x * self.scale
        x = x + self.bias4

        return original + x


class ResNetHead(nn.Module):
    """
    A small residual network CNN head for processing images.

    Architecture is from IMPALA paper in Fig 3 [1]
    """

    def __init__(self, in_channels=3, filter_sizes=(16, 32, 32), add_extra_block=False):
        super().__init__()
        self.num_total_blocks = len(filter_sizes) + int(add_extra_block)
        self.blocks = []

        # Scaler for FixUp mid-most convolutions.
        # Scaling is L^(-1/(2m - 2)) . In our case m = 2 (two layers in branch),
        # so our rescaling is L^(-1/2) = 1 / sqrt(L).
        # L is number of residual branches in our network.
        # Each block in IMPALA has two branches.
        first_conv_weight_scale = 1 / (math.sqrt(self.num_total_blocks * 2))
        input_channels = in_channels
        for output_channels in filter_sizes:
            block = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ResidualBlock(output_channels, first_conv_weight_scale),
                ResidualBlock(output_channels, first_conv_weight_scale),
            ]
            self.blocks.extend(block)
            input_channels = output_channels
        # Number of blocks without max pooling
        if add_extra_block:
            self.blocks.extend(
                (
                    ResidualBlock(output_channels, first_conv_weight_scale),
                    ResidualBlock(output_channels, first_conv_weight_scale),
                )
            )
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        x = F.relu(x, inplace=True)
        return x

    def get_embedding_dim(self):
        return self.encoder.get_embedding_dim()


class ImpalaResNet(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        assert len(image_shape) == 3, "ImpalaResNet expects 3 dimensions (channels, height, width)"
        in_channels = image_shape[0]
        self.encoder = ResNetHead(in_channels=in_channels)

        # Get output dimension of encoder by passing an example input through
        example_input = torch.randn(*image_shape)
        self.output_dim = self.encoder(example_input).reshape(-1).shape[0]

    def forward(self, x):
        assert x.dim() >= 3, "Input to encoder must be at least 3D (batch, channels, height, width)"
        out = self.encoder(x)
        # combine encoder dimensions of processed channels and height/width to embedding dimension
        return out.reshape(*out.shape[:-3], self.output_dim)

    def get_embedding_dim(self):
        return self.output_dim
