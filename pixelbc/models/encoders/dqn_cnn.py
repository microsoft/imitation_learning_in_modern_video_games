# source: https://github.com/Miffyli/minecraft-bc-2020/blob/master/torch_codes/modules.py
import torch
from torch import nn


class NatureDQNCNN(nn.Module):
    """The CNN head from Nature DQN paper"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(x)


class NatureCNN(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        assert len(img_shape) == 3, "NatureDQNCNN expects 3 dimensions (channels, height, width)"
        in_channels = img_shape[0]
        self.encoder = NatureDQNCNN(in_channels)

        # Get output dimension of encoder by passing an example input through
        example_input = torch.randn(*img_shape)
        self.output_dim = self.encoder(example_input).reshape(-1).shape[0]

    def forward(self, x):
        assert x.dim() >= 3, "Input to encoder must be at least 3D (batch, channels, height, width)"
        out = self.encoder(x)
        # combine encoder dimensions of processed channels and height/width to embedding dimension
        return out.reshape(*out.shape[:-3], self.output_dim)

    def get_embedding_dim(self):
        return self.output_dim
