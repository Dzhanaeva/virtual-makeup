from __future__ import annotations

import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LipSegmentationNet(nn.Module):
    """Small ROI model trained from scratch with no third-party weights."""

    def __init__(self, class_count: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = DepthwiseBlock(16, 24, stride=2)
        self.encoder2 = DepthwiseBlock(24, 40, stride=2)
        self.encoder3 = DepthwiseBlock(40, 64, stride=2)
        self.bottleneck = DepthwiseBlock(64, 80)
        self.decoder2 = DepthwiseBlock(80 + 40, 48)
        self.decoder1 = DepthwiseBlock(48 + 24, 32)
        self.decoder0 = DepthwiseBlock(32 + 16, 24)
        self.head = nn.Conv2d(24, class_count, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level0 = self.stem(inputs)
        level1 = self.encoder1(level0)
        level2 = self.encoder2(level1)
        level3 = self.encoder3(level2)
        features = self.bottleneck(level3)
        features = self._decode(features, level2, self.decoder2)
        features = self._decode(features, level1, self.decoder1)
        features = self._decode(features, level0, self.decoder0)
        return self.head(features)

    @staticmethod
    def _decode(
        features: torch.Tensor,
        skip: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        features = nn.functional.interpolate(
            features,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return block(torch.cat((features, skip), dim=1))

