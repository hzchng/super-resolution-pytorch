# Paper: https://arxiv.org/pdf/1609.04802
import math

from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2d_2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor):
        identity = x
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.prelu(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        return x + identity


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, kernel_size, stride, padding=1):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels * upscale_factor * upscale_factor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.ps = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor):
        x = self.conv2d(x)
        x = self.ps(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, blocks_depth=5, upscale_factor=2):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding="same"),
            nn.PReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, kernel_size=3, stride=1, padding="same") for _ in range(blocks_depth)]
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
        )

        blocks = []
        blocks.append(PixelShuffleBlock(64, 256, upscale_factor=2, kernel_size=3, stride=1, padding="same"))
        for _ in range(int(math.log(upscale_factor, 2)) - 1):
            blocks.append(PixelShuffleBlock(256, 256, upscale_factor=2, kernel_size=3, stride=1, padding="same"))
        self.upscale_blocks = nn.Sequential(*blocks)

        self.block_3 = nn.Conv2d(256, 3, kernel_size=9, stride=1, padding="same")

    def forward(self, x: Tensor):
        skip_1 = self.block_1(x)
        x = self.residual_blocks(skip_1)
        x = self.block_2(x)
        x = x + skip_1
        x = self.upscale_blocks(x)
        x = self.block_3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )
        self.block_2 = self.get_conv_block(64, 64, kernel_size=3, stride=2)
        self.block_3 = self.get_conv_block(64, 128, kernel_size=3, stride=1)
        self.block_4 = self.get_conv_block(128, 128, kernel_size=3, stride=2)
        self.block_5 = self.get_conv_block(128, 256, kernel_size=3, stride=1)
        self.block_6 = self.get_conv_block(256, 256, kernel_size=3, stride=2)
        self.block_7 = self.get_conv_block(256, 512, kernel_size=3, stride=1)
        self.block_8 = self.get_conv_block(512, 512, kernel_size=3, stride=2)

        self.block_9 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def get_conv_block(self, in_channels, out_channels, kernel_size, stride, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.block_9(x)
        return x
