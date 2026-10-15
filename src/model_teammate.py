"""Comparison CNN based on the original teammate draft."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeammateCNN(nn.Module):
    """Three convolution layers followed by four fully connected layers."""

    def __init__(self, num_classes: int = 3, img_size: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        flatten_size = self._get_flatten_size(img_size)

        self.fc1 = nn.Linear(flatten_size, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=7, stride=2, padding=3)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=7, stride=2, padding=3)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=7, stride=2, padding=3)
        return x

    def _get_flatten_size(self, img_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            features = self._forward_features(dummy)
            return int(features.view(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
