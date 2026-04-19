"""
SCDMN-ResNet18 for CIFAR-10.

Design notes:
- We apply one (gate, mask) per ResNet stage (layer1..layer4), NOT per BasicBlock.
  Rationale: channel counts change between stages, and stage-level granularity
  is enough to observe the shallow->deep gate pattern (our core hypothesis).
- Context encoder modes:
    'cnn'    : small CNN reads the raw image -> z
    'oracle' : embedding table keyed by ground-truth context label -> z
- The plain ResNet18 baseline lives in models/resnet_baseline.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scdmn_components import (
    ContextEncoderCNN,
    ContextEncoderOracle,
    MaskGenerator,
    GatedMaskApply,
)


# -------- ResNet18 building blocks (CIFAR variant) --------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


def _make_stage(in_planes, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
        layers.append(BasicBlock(in_planes, planes, s))
        in_planes = planes * BasicBlock.expansion
    return nn.Sequential(*layers), in_planes


# -------- SCDMN-ResNet18 --------

class SCDMNResNet18(nn.Module):
    """
    ResNet18 (CIFAR variant) with scene-conditional channel masking
    applied at the output of each of the 4 stages.

    Args:
        num_classes: number of classification targets
        num_contexts: number of context labels (for oracle mode)
        context_mode: 'cnn' or 'oracle'
        z_dim: context embedding dimensionality
        sparsity: fraction of channels kept in the mask (0.5 = 50%)
    """
    def __init__(
        self,
        num_classes: int = 10,
        num_contexts: int = 4,
        context_mode: str = 'cnn',
        z_dim: int = 128,
        sparsity: float = 0.5,
    ):
        super().__init__()
        assert context_mode in ('cnn', 'oracle')
        self.context_mode = context_mode

        # CIFAR stem: 3x3 conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 stages like ResNet34: [3,4,6,3], channels [64,128,256,512]
        self.layer1, c1 = _make_stage(64,  64,  3, stride=1)
        self.layer2, c2 = _make_stage(c1, 128, 4, stride=2)
        self.layer3, c3 = _make_stage(c2, 256, 6, stride=2)
        self.layer4, c4 = _make_stage(c3, 512, 3, stride=2)
        self.stage_channels = [c1, c2, c3, c4]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

        # Context encoder
        if context_mode == 'cnn':
            self.ctx_encoder = ContextEncoderCNN(in_channels=3, z_dim=z_dim)
        else:
            self.ctx_encoder = ContextEncoderOracle(num_contexts=num_contexts, z_dim=z_dim)

        # Mask generator (one head per stage) and one gate per stage
        self.mask_gen = MaskGenerator(
            z_dim=z_dim,
            layer_channels=self.stage_channels,
            sparsity=sparsity,
        )
        self.gates = nn.ModuleList([GatedMaskApply(init_gate_logit=-2.0) for _ in self.stage_channels])

        # Cache for analysis
        self._last_masks = None

    # ---- analysis utilities ----

    @torch.no_grad()
    def get_gate_values(self):
        """Returns list of current gate values (sigmoid of raw logits)."""
        return [g.gate.item() for g in self.gates]

    def get_last_masks(self):
        """Returns list of masks from the last forward pass (detached)."""
        if self._last_masks is None:
            return None
        return [m.detach() for m in self._last_masks]

    # ---- forward ----

    def _encode_context(self, x: torch.Tensor, ctx_label: torch.Tensor = None) -> torch.Tensor:
        if self.context_mode == 'cnn':
            return self.ctx_encoder(x)
        else:
            if ctx_label is None:
                raise ValueError("context_mode='oracle' requires ctx_label.")
            return self.ctx_encoder(ctx_label)

    def forward(
        self,
        x: torch.Tensor,
        ctx_label: torch.Tensor = None,
        return_features: bool = False,
        hard_mask: bool = True,
    ):
        """
        Args:
            x: (B, 3, 32, 32)
            ctx_label: (B,) long tensor, required if context_mode='oracle'
            return_features: if True, returns list of features after each masked stage
                             (used for Linear Probe analysis)
            hard_mask: True = Top-k STE, False = soft sigmoid (warmup / debugging)
        """
        z = self._encode_context(x, ctx_label)                       # (B, z_dim)
        masks = self.mask_gen(z, hard=hard_mask)                     # list of (B, C_i)
        self._last_masks = masks

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        features = []
        for stage, gate_mod, mask in zip(
            [self.layer1, self.layer2, self.layer3, self.layer4],
            self.gates,
            masks,
        ):
            out = stage(out)
            out = gate_mod(out, mask)
            if return_features:
                features.append(out)

        out = self.pool(out).flatten(1)
        logits = self.fc(out)

        if return_features:
            return logits, features
        return logits