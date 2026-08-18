import torch.nn as nn


class RGBDecoder(nn.Module):
    """Reconstructs an RGB frame from a frozen image-encoder embedding.

    Auxiliary task: projects the embedding to a small spatial feature map, then
    upsamples with transposed convolutions to the target resolution. Output is
    in [0, 1] (plain pixel space, not ImageNet-normalised), so the reconstruction
    loss is a direct MSE against the un-normalised input frame.
    """

    def __init__(self, embedding_dim, output_size=224, output_channels=3, base_channels=512):
        super().__init__()
        assert output_size % 32 == 0, f"output_size must be a multiple of 32, got {output_size}"
        self.start_size = output_size // 32
        self.base_channels = base_channels

        self.project = nn.Linear(embedding_dim, base_channels * self.start_size * self.start_size)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            up_block(base_channels, base_channels // 2),  # start_size -> 2x
            up_block(base_channels // 2, base_channels // 4),  # -> 4x
            up_block(base_channels // 4, base_channels // 8),  # -> 8x
            up_block(base_channels // 8, base_channels // 16),  # -> 16x
            nn.ConvTranspose2d(base_channels // 16, output_channels, kernel_size=4, stride=2, padding=1),  # -> 32x
            nn.Sigmoid(),
        )

    def forward(self, embedding):
        """
        :param embedding: (batch, embedding_dim)
        :return: (batch, output_channels, output_size, output_size) in [0, 1]
        """
        x = self.project(embedding)
        x = x.reshape(-1, self.base_channels, self.start_size, self.start_size)
        return self.decoder(x)
