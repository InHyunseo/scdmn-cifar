"""
Plain ResNet18 for CIFAR-10. Used as:
- Baseline 1: single model trained on all contexts mixed.
- Baseline 2: one instance per context (independent experts, oracle routing).

Same CIFAR stem (3x3, no maxpool) as SCDMN for fair comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scdmn_resnet import BasicBlock, _make_stage


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1, c1 = _make_stage(64,  64,  3, stride=1)
        self.layer2, c2 = _make_stage(c1, 128, 4, stride=2)
        self.layer3, c3 = _make_stage(c2, 256, 6, stride=2)
        self.layer4, c4 = _make_stage(c3, 512, 3, stride=2)
        self.stage_channels = [c1, c2, c3, c4]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        features = []
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = stage(out)
            if return_features:
                features.append(out)

        out = self.pool(out).flatten(1)
        logits = self.fc(out)
        if return_features:
            return logits, features
        return logits


class IndependentExperts(nn.Module):
    """
    One ResNet18 per context, with oracle routing.
    Serves as an upper bound: full parameter separation, correct context known.
    """
    def __init__(self, num_classes: int = 10, num_contexts: int = 4):
        super().__init__()
        self.num_contexts = num_contexts
        self.experts = nn.ModuleList([
            ResNet18CIFAR(num_classes=num_classes) for _ in range(num_contexts)
        ])

    def forward(self, x: torch.Tensor, ctx_label: torch.Tensor):
        """
        ctx_label: (B,) long tensor in [0, num_contexts).
        We group the batch by context, run each expert on its slice, then scatter back.
        """
        B = x.size(0)
        out_shape = None
        logits_buffer = None

        for c in range(self.num_contexts):
            mask = (ctx_label == c)
            if not mask.any():
                continue
            sub_x = x[mask]
            sub_logits = self.experts[c](sub_x)
            if logits_buffer is None:
                out_shape = sub_logits.shape[1:]
                logits_buffer = x.new_zeros((B,) + out_shape)
            logits_buffer[mask] = sub_logits

        if logits_buffer is None:
            # Safety fallback: empty batch per context (shouldn't happen in practice)
            logits_buffer = self.experts[0](x)
        return logits_buffer