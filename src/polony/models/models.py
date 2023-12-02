from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(
        self,
        channels: Tuple[int, int],
        size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        N: int = 1,
        res: bool = False,
    ):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(channels, size, stride, N),
        )
        self.res = res
        if self.res:
            self.conv_skip = nn.Conv2d(*channels, kernel_size=1)

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        if self.res:
            to_upsample = self.conv(to_conv) + self.conv_skip(to_conv)
        else:
            to_upsample = self.conv(to_conv)

        upsample_conv = F.interpolate(to_upsample, size=to_cat.shape[-2:])
        return torch.cat([upsample_conv, to_cat], dim=1)


def conv_block(
    channels: Tuple[int, int],
    size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    N: int = 1,
):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """

    # a single convolution + batch normalization + ReLU block
    def block(in_channels: int):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels[1],
                kernel_size=size,
                stride=stride,
                bias=False,
                padding=(size[0] // 2, size[1] // 2),
            ),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(),
        )

    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class UNet(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(
        self,
        filters: int = 64,
        input_filters: int = 2,
        N: int = 2,
        res: bool = False,
        **kwargs
    ):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        self.res = res

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=N)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=N)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=N)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=N, res=res)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=N, res=res)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=N, res=res)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=N)
        self.density_pred = nn.Conv2d(
            in_channels=filters, out_channels=1, kernel_size=(1, 1), bias=False
        )

        # residual
        if self.res:
            self.conv_skip1 = nn.Conv2d(*initial_filters, kernel_size=1)
            self.conv_skip2 = nn.Conv2d(*down_filters, kernel_size=1)
            self.conv_skip3 = nn.Conv2d(*down_filters, kernel_size=1)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        if self.res:
            block1 = self.block1(input) + self.conv_skip1(input)
        else:
            block1 = self.block1(input)
        pool1 = pool(block1)
        if self.res:
            block2 = self.block2(pool1) + self.conv_skip2(pool1)
        else:
            block2 = self.block2(pool1)
        pool2 = pool(block2)
        if self.res:
            block3 = self.block3(pool2) + self.conv_skip3(pool2)
        else:
            block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)


class Classifier(nn.Module):
    def __init__(self, in_channel: int = 2) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.weights = ResNet18_Weights.DEFAULT
        self.net = resnet18(weights=self.weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)
        if self.in_channel != 3:
            self.net.conv1 = nn.Conv2d(
                self.in_channel,
                self.net.conv1.out_channels,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.net(inp)

    def freeze_layers(self, mode: str = "on") -> None:
        if mode == "on":
            for param in self.parameters():
                param.requires_grad_ = False
            self.net.fc.requires_grad_ = True
            if self.in_channel != 3:
                self.net.conv1.requires_grad_ = True
        elif mode == "off":
            for param in self.parameters():
                param.requires_grad_ = True
        else:
            raise ValueError("Wrong mode, should be 'on' or 'off'")
