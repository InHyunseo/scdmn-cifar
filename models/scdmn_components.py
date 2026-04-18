"""
SCDMN core components:
- ContextEncoder: two modes (CNN from image, Oracle from label)
- MaskGenerator: per-layer channel-wise mask with Top-k STE
- GatedMaskApply: applies gate + mask to activations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSTE(torch.autograd.Function):
    """
    Top-k Straight-Through Estimator.
    Forward: keeps top-k channels (by score), sets rest to 0, scales kept to 1.
    Backward: passes gradient through as if identity on the sigmoid output.
    """
    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int):
        # scores: (B, C), k: int
        # Pick top-k indices per batch row
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient flows to scores unchanged
        return grad_output, None


def topk_ste(scores: torch.Tensor, k: int) -> torch.Tensor:
    return TopKSTE.apply(scores, k)


class ContextEncoderCNN(nn.Module):
    """
    Small CNN that maps raw image -> context embedding z.
    Kept small (~50k params) so it doesn't dominate the main network.
    """
    def __init__(self, in_channels: int = 3, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # 16 -> 8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 8 -> 4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextEncoderOracle(nn.Module):
    """
    Oracle encoder: takes integer context label, outputs learned embedding.
    Used as an upper bound to isolate 'is the mask structure working?'
    from 'can we infer context from the image?'.
    """
    def __init__(self, num_contexts: int, z_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_contexts, z_dim)

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        # context_labels: (B,) long tensor
        return self.embed(context_labels)


class MaskGenerator(nn.Module):
    """
    Maps z -> per-layer channel-wise binary mask (via Top-k STE).
    Each layer has its own head so masks at different depths are independent.

    Args:
        z_dim: context embedding dim
        layer_channels: list of channel counts for each masked layer (e.g. [64, 128, 256, 512])
        sparsity: fraction of channels to KEEP (e.g. 0.5 = keep 50%)
        hidden_dim: MLP hidden dim for each head
    """
    def __init__(
        self,
        z_dim: int,
        layer_channels: list,
        sparsity: float = 0.5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.layer_channels = layer_channels
        self.sparsity = sparsity
        self.keep_counts = [max(1, int(c * sparsity)) for c in layer_channels]

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, c),
            )
            for c in layer_channels
        ])

    def forward(self, z: torch.Tensor, hard: bool = True) -> list:
        """
        Returns list of masks, one per layer, each of shape (B, C_layer).
        If hard=True: Top-k binary mask with STE.
        If hard=False: soft sigmoid (for warmup / debugging).
        """
        masks = []
        for head, keep_k in zip(self.heads, self.keep_counts):
            logits = head(z)                # (B, C)
            scores = torch.sigmoid(logits)  # (B, C) in [0,1]
            if hard:
                mask = topk_ste(scores, keep_k)
                # Rescale so expected value of each kept channel is preserved
                # (optional; we use /sparsity for unbiased activation magnitude)
                mask = mask / self.sparsity
            else:
                mask = scores / self.sparsity
            masks.append(mask)
        return masks


class GatedMaskApply(nn.Module):
    """
    Applies gate + mask to an activation tensor of shape (B, C, H, W).

    Formula:   out = x * (1 - g + g * mask)
    where g is a learnable scalar in [0,1] (sigmoid of a raw param).

    - g -> 0: mask ignored, full sharing
    - g -> 1: mask fully applied, context-specific routing
    """
    def __init__(self, init_gate_logit: float = -2.0):
        super().__init__()
        # Start with g ~ sigmoid(-2) ~ 0.12, so early training is mostly shared.
        # Model can learn to raise it where context separation helps.
        self.gate_logit = nn.Parameter(torch.tensor(init_gate_logit))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x:    (B, C, H, W)
        # mask: (B, C)
        g = self.gate
        mask_bcHW = mask.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        effective = (1.0 - g) + g * mask_bcHW          # (B, C, 1, 1)
        return x * effective