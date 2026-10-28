from __future__ import annotations

from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_gesture_recognition.pipeline.models.neural.base import ClassifierModule


class ParallelConvLayer(nn.Module):
    """Inception-style parallel conv layer: four branches concatenated on channels.

    Branches: 1x1->ReLU->3x3, 1x1, 3x3 MaxPool(stride=1)->1x1, 1x1->ReLU->5x5.
    All preserve spatial dims; output channels = 4 * out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 16,
        out_channels: int = 18,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class FeatureExtractionModule(nn.Module):
    """Stacked parallel-conv feature extractor with residual connections + LRN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 18,
        lrn_size: int = 5,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ):
        super().__init__()
        width = 4 * out_channels
        self.block1 = ParallelConvLayer(
            in_channels=in_channels, out_channels=out_channels
        )
        self.block2 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block3 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block4 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block5 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block6 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.cross_channel_norm = nn.LocalResponseNorm(
            size=lrn_size, alpha=alpha, beta=beta, k=k
        )
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.block1(x))
        res1 = x
        x = F.relu(self.block2(x))
        x = self.block3(x)
        x = F.relu(x + res1)
        res2 = x
        x = F.relu(self.block4(x))
        x = self.block5(x)
        x = F.relu(x + res2)
        x = self.block6(x)
        return self.cross_channel_norm(x)


class ParallelConvModule(ClassifierModule):
    """2D parallel-conv feature extractor + FC head.

    Input ``(batch, channels, small_windows, features)``; output logits
    ``(batch, n_classes)``. The shared ``ClassifierModule`` base provides
    cross-entropy loss, argmax accuracy, macro-F1, optimizer, and early-stopping.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        out_channels: int = 18,
        optimizer: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_classes=n_classes, optimizer=optimizer)
        self.features = FeatureExtractionModule(
            in_channels=n_channels, out_channels=out_channels
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (N, 4*out_channels, H, W)
        x = self.pool(x).flatten(1)  # (N, 4*out_channels)
        return self.head(x)  # (N, n_classes)
