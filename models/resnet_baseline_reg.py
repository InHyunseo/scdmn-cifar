"""
Single-model regression baseline: ResNet18 CIFAR with 1-output tanh head.
Ignores context label (trained on mixed contexts).
"""
import torch
import torch.nn as nn

from .resnet_baseline import ResNet18CIFAR


class BaselineResNetReg(nn.Module):
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.net = ResNet18CIFAR(num_classes=1, width_mult=width_mult)

    def forward(self, x: torch.Tensor, ctx_label=None) -> torch.Tensor:
        return torch.tanh(self.net(x))
