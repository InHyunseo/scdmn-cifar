"""
SCDMN-Sliced: oracle-only variant where per-context channel masks actually
slice the conv/BN tensors so FLOPs decrease, not just activations get zeroed.

Design:
- Per-context channel scores s_c ∈ R^C per stage (learned embedding).
- Forward is ALWAYS context-grouped (batch split by ctx_label, run each
  context's sub-network, scatter back). Soft and sliced modes share this
  structure so train/eval distributions match.
- Soft mode (warmup): mask = sigmoid(s_c) ∈ (0,1)^C, applied to
  conv WEIGHTS along the output-channel axis (and to BN affine/stats
  along the channel axis). Full-width conv runs but channels with low
  score contribute ~0 to both forward and gradient.
- Freeze: top-k channels by score per (ctx, stage) become a binary mask
  / index list. After freeze, sliced mode runs only those channels —
  identical computation to soft mode at the limit where mask -> {0,1}.
- Stem (conv1/bn1) and fc are shared full-width across contexts; only
  fc input is sliced per context using the last stage's active set.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- low-level sliced ops (used post-freeze) ----------

def _sliced_bn(
    x: torch.Tensor,
    bn: nn.BatchNorm2d,
    idx: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    w = bn.weight[idx] if bn.affine else None
    b = bn.bias[idx] if bn.affine else None
    if training and bn.track_running_stats:
        rm = bn.running_mean[idx].clone()
        rv = bn.running_var[idx].clone()
        out = F.batch_norm(x, rm, rv, weight=w, bias=b,
                           training=True, momentum=bn.momentum, eps=bn.eps)
        with torch.no_grad():
            bn.running_mean.index_copy_(0, idx, rm)
            bn.running_var.index_copy_(0, idx, rv)
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)
        return out
    else:
        rm = bn.running_mean[idx]
        rv = bn.running_var[idx]
        return F.batch_norm(x, rm, rv, weight=w, bias=b,
                            training=False, momentum=bn.momentum, eps=bn.eps)


def _sliced_conv(
    x: torch.Tensor,
    conv: nn.Conv2d,
    out_idx: torch.Tensor,
    in_idx: torch.Tensor,
) -> torch.Tensor:
    w = conv.weight.index_select(0, out_idx).index_select(1, in_idx)
    bias = conv.bias.index_select(0, out_idx) if conv.bias is not None else None
    return F.conv2d(x, w, bias=bias,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups)


# ---------- soft (weight-masked) ops (used during warmup) ----------

def _soft_conv(
    x: torch.Tensor,
    conv: nn.Conv2d,
    m_out: torch.Tensor,    # (C_out,) in (0,1)
    m_in: torch.Tensor,     # (C_in,)  in (0,1)
) -> torch.Tensor:
    """conv with weight scaled by m_out on out-axis and m_in on in-axis."""
    w = conv.weight * m_out.view(-1, 1, 1, 1) * m_in.view(1, -1, 1, 1)
    return F.conv2d(x, w, bias=conv.bias,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups)


def _soft_bn(
    x: torch.Tensor,
    bn: nn.BatchNorm2d,
    m: torch.Tensor,        # (C,) in (0,1)
    training: bool,
) -> torch.Tensor:
    """
    BN with affine weight scaled by m. Running stats are updated in-place
    over the full channel dim (channels with m~0 will see near-zero activations,
    which is fine — their stats are valid but unused after slicing).
    """
    out = F.batch_norm(
        x, bn.running_mean, bn.running_var,
        weight=(bn.weight * m) if bn.affine else None,
        bias=bn.bias if bn.affine else None,
        training=training and bn.track_running_stats,
        momentum=bn.momentum, eps=bn.eps,
    )
    if bn.num_batches_tracked is not None and training and bn.track_running_stats:
        # F.batch_norm with running_mean/var passed in already updated them; no extra step needed.
        pass
    return out


# ---------- block ----------

class SlicedBasicBlock(nn.Module):
    """
    Two forward variants — both follow the same arithmetic, differing only
    in how channel selection is realized (soft scaling vs hard indexing).
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.has_proj = (stride != 1) or (in_planes != planes * self.expansion)
        if self.has_proj:
            self.proj_conv = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)
            self.proj_bn = nn.BatchNorm2d(planes * self.expansion)

    # --- soft (weight-masked) ---

    def forward_soft(
        self,
        x: torch.Tensor,
        m_in: torch.Tensor,    # (C_in,)
        m_mid: torch.Tensor,   # (C_mid,)
        m_out: torch.Tensor,   # (C_out,)
        training: bool,
    ) -> torch.Tensor:
        out = _soft_conv(x, self.conv1, m_mid, m_in)
        out = _soft_bn(out, self.bn1, m_mid, training)
        out = F.relu(out, inplace=True)

        out = _soft_conv(out, self.conv2, m_out, m_mid)
        out = _soft_bn(out, self.bn2, m_out, training)

        if self.has_proj:
            sc = _soft_conv(x, self.proj_conv, m_out, m_in)
            sc = _soft_bn(sc, self.proj_bn, m_out, training)
        else:
            # Identity path: scale x by m_out so its effective channels
            # match those of the residual branch. Requires m_in == m_out
            # at no-proj boundaries in expectation; here both equal stage mask.
            sc = x * m_out.view(1, -1, 1, 1)
        return F.relu(out + sc, inplace=True)

    # --- hard sliced (post-freeze) ---

    def forward_sliced(
        self,
        x: torch.Tensor,
        in_idx: torch.Tensor,
        mid_idx: torch.Tensor,
        out_idx: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        out = _sliced_conv(x, self.conv1, mid_idx, in_idx)
        out = _sliced_bn(out, self.bn1, mid_idx, training)
        out = F.relu(out, inplace=True)

        out = _sliced_conv(out, self.conv2, out_idx, mid_idx)
        out = _sliced_bn(out, self.bn2, out_idx, training)

        if self.has_proj:
            sc = _sliced_conv(x, self.proj_conv, out_idx, in_idx)
            sc = _sliced_bn(sc, self.proj_bn, out_idx, training)
        else:
            if in_idx is out_idx or (in_idx.shape == out_idx.shape and torch.equal(in_idx, out_idx)):
                sc = x
            else:
                # in_idx ⊃ out_idx (e.g., stage 0: in=full stem, out=subset).
                if in_idx.numel() == in_idx.max().item() + 1 and in_idx[0].item() == 0:
                    pos = out_idx
                else:
                    pos = torch.searchsorted(in_idx, out_idx)
                sc = x.index_select(1, pos)
        return F.relu(out + sc, inplace=True)


# ---------- main model ----------

class SCDMNSliced(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_contexts: int = 4,
        sparsity: float = 0.25,
        stage_blocks=(3, 4, 6, 3),
        stage_channels=(64, 128, 256, 512),
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.sparsity = sparsity
        self.stage_channels = list(stage_channels)
        self.keep_counts = [max(1, int(round(c * sparsity))) for c in self.stage_channels]

        self.conv1 = nn.Conv2d(3, self.stage_channels[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.stage_channels[0])

        self.stages = nn.ModuleList()
        in_planes = self.stage_channels[0]
        for stage_idx, (planes, n_blocks) in enumerate(zip(self.stage_channels, stage_blocks)):
            stride = 1 if stage_idx == 0 else 2
            blocks = nn.ModuleList()
            strides = [stride] + [1] * (n_blocks - 1)
            for s in strides:
                blocks.append(SlicedBasicBlock(in_planes, planes, s))
                in_planes = planes
            self.stages.append(blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage_channels[-1], num_classes)

        # Per-context channel scores. Larger init std so contexts diverge faster.
        self.channel_scores = nn.ParameterList([
            nn.Parameter(torch.randn(num_contexts, c) * 0.5)
            for c in self.stage_channels
        ])

        self._frozen = False
        for i, k in enumerate(self.keep_counts):
            self.register_buffer(
                f"frozen_idx_{i}",
                torch.zeros(num_contexts, k, dtype=torch.long),
                persistent=True,
            )

    # ---- mask freezing ----

    def freeze_masks(self):
        with torch.no_grad():
            for i, scores in enumerate(self.channel_scores):
                k = self.keep_counts[i]
                _, idx = torch.topk(scores, k, dim=-1)
                idx, _ = torch.sort(idx, dim=-1)
                getattr(self, f"frozen_idx_{i}").copy_(idx)
        self._frozen = True

    def is_frozen(self) -> bool:
        return self._frozen

    def get_active_idx(self, stage_i: int, ctx: int) -> torch.Tensor:
        return getattr(self, f"frozen_idx_{stage_i}")[ctx]

    # ---- forward ----

    def forward(self, x: torch.Tensor, ctx_label: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        B = out.size(0)
        device = out.device
        logits = out.new_zeros((B, self.fc.out_features))

        full_in = torch.arange(self.stage_channels[0], device=device)

        for c in range(self.num_contexts):
            sel = (ctx_label == c)
            if not sel.any():
                continue
            sub = out[sel]

            if self._frozen:
                # Sliced path
                stage_idxs = [self.get_active_idx(i, c).to(device) for i in range(len(self.stages))]
                prev_idx = full_in
                for stage_i, blocks in enumerate(self.stages):
                    cur_idx = stage_idxs[stage_i]
                    for block_j, block in enumerate(blocks):
                        in_idx = prev_idx if block_j == 0 else cur_idx
                        sub = block.forward_sliced(
                            sub, in_idx=in_idx, mid_idx=cur_idx, out_idx=cur_idx,
                            training=self.training,
                        )
                    prev_idx = cur_idx
                feat = self.pool(sub).flatten(1)
                w = self.fc.weight.index_select(1, prev_idx)
                sub_logits = F.linear(feat, w, self.fc.bias)
            else:
                # Soft path: weight-masked, full width
                # Stage-0 block-0 input mask = ones (stem is shared full-width).
                m_prev = torch.ones(self.stage_channels[0], device=device, dtype=sub.dtype)
                for stage_i, blocks in enumerate(self.stages):
                    m_cur = torch.sigmoid(self.channel_scores[stage_i][c])  # (C_stage,)
                    for block_j, block in enumerate(blocks):
                        m_in = m_prev if block_j == 0 else m_cur
                        sub = block.forward_soft(
                            sub, m_in=m_in, m_mid=m_cur, m_out=m_cur,
                            training=self.training,
                        )
                    m_prev = m_cur
                feat = self.pool(sub).flatten(1)
                # fc input mask = last stage's mask
                fc_w = self.fc.weight * m_prev.view(1, -1)
                sub_logits = F.linear(feat, fc_w, self.fc.bias)

            logits[sel] = sub_logits

        return logits

    # ---- analysis ----

    @torch.no_grad()
    def mask_iou_matrix(self, stage_i: int) -> torch.Tensor:
        assert self._frozen, "freeze_masks() first"
        idx = getattr(self, f"frozen_idx_{stage_i}")
        n = idx.size(0)
        iou = torch.zeros(n, n)
        for i in range(n):
            si = set(idx[i].tolist())
            for j in range(n):
                sj = set(idx[j].tolist())
                inter = len(si & sj)
                union = len(si | sj)
                iou[i, j] = inter / union if union > 0 else 0.0
        return iou
