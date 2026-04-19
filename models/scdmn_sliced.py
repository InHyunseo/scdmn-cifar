"""
SCDMN-Sliced: oracle-only variant where per-context channel masks actually
slice the conv/BN tensors so FLOPs decrease, not just activations get zeroed.

Key differences from SCDMN:
- Channel scores are per-context embeddings (context x stage), not per-sample.
- Warmup phase: soft sigmoid mask multiplied on full-width conv output
  (so all weights receive gradient, like the original SCDMN warmup).
- After mask_freeze_epoch: top-k indices are frozen per context. Forward
  becomes a true sliced forward — for each context group in the batch,
  conv weight is indexed to [out_idx][:, in_idx] and BN params likewise.
- Stem (conv1/bn1) and fc are shared full-width across contexts.
- Stage k's output active set == Stage (k+1)'s input active set, so each
  context defines a coherent sub-network through the residual stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- helpers ----------

def _sliced_bn(
    x: torch.Tensor,
    bn: nn.BatchNorm2d,
    idx: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """
    Run F.batch_norm on the slice [:, idx] using the corresponding sliced
    weight/bias/running stats. Updates the original BN buffers in-place via
    index_copy_ when training, so stats accumulate correctly.
    """
    w = bn.weight[idx] if bn.affine else None
    b = bn.bias[idx] if bn.affine else None

    if training and bn.track_running_stats:
        rm = bn.running_mean[idx].clone()
        rv = bn.running_var[idx].clone()
        out = F.batch_norm(
            x, rm, rv, weight=w, bias=b,
            training=True, momentum=bn.momentum, eps=bn.eps,
        )
        with torch.no_grad():
            bn.running_mean.index_copy_(0, idx, rm)
            bn.running_var.index_copy_(0, idx, rv)
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)
        return out
    else:
        rm = bn.running_mean[idx]
        rv = bn.running_var[idx]
        return F.batch_norm(
            x, rm, rv, weight=w, bias=b,
            training=False, momentum=bn.momentum, eps=bn.eps,
        )


def _sliced_conv(
    x: torch.Tensor,
    conv: nn.Conv2d,
    out_idx: torch.Tensor,
    in_idx: torch.Tensor,
) -> torch.Tensor:
    w = conv.weight.index_select(0, out_idx).index_select(1, in_idx)
    bias = conv.bias.index_select(0, out_idx) if conv.bias is not None else None
    return F.conv2d(
        x, w, bias=bias,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
    )


# ---------- sliced BasicBlock ----------

class SlicedBasicBlock(nn.Module):
    """
    BasicBlock with channel slicing on every conv/BN.
    Soft path (warmup): runs full-width conv, multiplies a soft mask on output.
    Sliced path (post-freeze): runs only the active channels.

    Active set semantics:
      - in_idx: input channels active for this block (= prev block's out_idx,
                or the stage's input active set for the first block).
      - mid_idx: hidden channels active inside this block.
      - out_idx: output channels active for this block (= mid_idx for BasicBlock,
                 since expansion=1 and we tie the stage's active set per context).
    For block-internal slicing we use mid_idx for both conv1 out and conv2 in/out.
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

    # ----- forward variants -----

    def forward_soft(
        self,
        x: torch.Tensor,
        soft_in: torch.Tensor,    # (in_planes,) in [0,1] or scaled
        soft_mid: torch.Tensor,   # (planes,)
        soft_out: torch.Tensor,   # (planes,)
    ) -> torch.Tensor:
        """Full-width conv with soft channel masks multiplied on activations."""
        # Mask input first (so input-channel sparsity is also exercised)
        x_in = x * soft_in.view(1, -1, 1, 1)
        out = F.relu(self.bn1(self.conv1(x_in)), inplace=True)
        out = out * soft_mid.view(1, -1, 1, 1)
        out = self.bn2(self.conv2(out))
        if self.has_proj:
            sc = self.proj_bn(self.proj_conv(x_in))
        else:
            sc = x_in
        out = out + sc
        out = out * soft_out.view(1, -1, 1, 1)
        return F.relu(out, inplace=True)

    def forward_sliced(
        self,
        x: torch.Tensor,          # already sliced to in_idx along dim=1
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
            # Identity path: x is in in_idx channels, out is in out_idx channels.
            # If they match (in_idx is out_idx), use directly.
            # Otherwise (e.g., stage 0 first block: in_idx = full stem, out_idx = active subset),
            # gather x along channel dim using positions of out_idx within in_idx.
            if in_idx is out_idx or (in_idx.shape == out_idx.shape and torch.equal(in_idx, out_idx)):
                sc = x
            else:
                # Build positional index: where each out_idx element sits inside in_idx.
                # Assumes out_idx ⊆ in_idx (true when in_idx is full stem range and out_idx is a subset).
                # Fast path: if in_idx is a full arange starting at 0, positions == out_idx values.
                if in_idx.numel() == in_idx.max().item() + 1 and in_idx[0].item() == 0:
                    pos = out_idx
                else:
                    # Generic: search positions
                    pos = torch.searchsorted(in_idx, out_idx)
                sc = x.index_select(1, pos)
        out = out + sc
        return F.relu(out, inplace=True)


# ---------- main model ----------

class SCDMNSliced(nn.Module):
    """
    Oracle-only SCDMN variant with true channel slicing.

    Args:
        num_classes
        num_contexts
        sparsity: fraction of channels to keep per stage per context
        stage_blocks: blocks per stage (default ResNet34: [3,4,6,3])
        stage_channels: full channel width per stage (default [64,128,256,512])
    """
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

        # Shared stem
        self.conv1 = nn.Conv2d(3, self.stage_channels[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.stage_channels[0])

        # Build stages
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
        # fc takes the LAST stage's full width; we slice along input dim per context.
        self.fc = nn.Linear(self.stage_channels[-1], num_classes)

        # Per-context channel scores: one (num_contexts, C) param per stage.
        # Stage k's score selects active channels for stage k's blocks (mid + out).
        self.channel_scores = nn.ParameterList([
            nn.Parameter(torch.randn(num_contexts, c) * 0.01)
            for c in self.stage_channels
        ])

        # Frozen index buffers (filled at freeze time): one (num_contexts, k) per stage.
        # When None, we are in soft (warmup) mode.
        self._frozen = False
        for i, k in enumerate(self.keep_counts):
            self.register_buffer(
                f"frozen_idx_{i}",
                torch.zeros(num_contexts, k, dtype=torch.long),
                persistent=True,
            )

    # ---- mask freezing ----

    def freeze_masks(self):
        """Freeze top-k indices per (context, stage) from current scores."""
        with torch.no_grad():
            for i, scores in enumerate(self.channel_scores):
                k = self.keep_counts[i]
                _, idx = torch.topk(scores, k, dim=-1)   # (num_contexts, k)
                idx, _ = torch.sort(idx, dim=-1)         # sort for deterministic ordering
                getattr(self, f"frozen_idx_{i}").copy_(idx)
        self._frozen = True

    def is_frozen(self) -> bool:
        return self._frozen

    def get_active_idx(self, stage_i: int, ctx: int) -> torch.Tensor:
        return getattr(self, f"frozen_idx_{stage_i}")[ctx]

    # ---- soft (warmup) helpers ----

    def _soft_mask(self, stage_i: int, ctx_label: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample soft mask of shape (B, C_stage), scaled so the
        expected magnitude matches the binary case ( /sparsity ).
        """
        scores = self.channel_scores[stage_i]                   # (num_ctx, C)
        soft = torch.sigmoid(scores)                            # (num_ctx, C)
        # Top-k STE-like behavior in soft mode: just scale; binary is post-freeze
        return soft[ctx_label] / self.sparsity                  # (B, C)

    # ---- forward ----

    def forward(self, x: torch.Tensor, ctx_label: torch.Tensor) -> torch.Tensor:
        # Stem (shared, full width)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        if not self._frozen:
            return self._forward_soft(out, ctx_label)
        return self._forward_sliced(out, ctx_label)

    def _forward_soft(self, out: torch.Tensor, ctx_label: torch.Tensor) -> torch.Tensor:
        """Per-sample soft mask multiplied on full-width activations."""
        B = out.size(0)
        # Stage input active set for stage 0 = full stem width (no mask).
        prev_mask = torch.ones(B, self.stage_channels[0], device=out.device, dtype=out.dtype)
        for stage_i, blocks in enumerate(self.stages):
            stage_mask = self._soft_mask(stage_i, ctx_label)    # (B, C_stage)
            for block_j, block in enumerate(blocks):
                in_mask = prev_mask if block_j == 0 else stage_mask
                # mid and out both use stage_mask (tie inside the stage)
                # Use per-channel average across batch as the soft scalar mask
                # (we want a (C,) mask per sample; here we apply per-sample).
                out = self._block_soft_per_sample(block, out, in_mask, stage_mask, stage_mask)
                prev_mask = stage_mask
            # End of stage: input to next stage's first block uses this stage_mask
        # Pool + fc (full width; fc input is masked stage4 output, but channels
        # not in active set are ~zeroed by the soft mask scaling)
        feat = self.pool(out).flatten(1)
        return self.fc(feat)

    @staticmethod
    def _block_soft_per_sample(
        block: SlicedBasicBlock,
        x: torch.Tensor,
        m_in: torch.Tensor,    # (B, C_in)
        m_mid: torch.Tensor,   # (B, C_mid)
        m_out: torch.Tensor,   # (B, C_out)
    ) -> torch.Tensor:
        x_in = x * m_in.unsqueeze(-1).unsqueeze(-1)
        out = F.relu(block.bn1(block.conv1(x_in)), inplace=True)
        out = out * m_mid.unsqueeze(-1).unsqueeze(-1)
        out = block.bn2(block.conv2(out))
        if block.has_proj:
            sc = block.proj_bn(block.proj_conv(x_in))
        else:
            sc = x_in
        out = out + sc
        out = out * m_out.unsqueeze(-1).unsqueeze(-1)
        return F.relu(out, inplace=True)

    def _forward_sliced(self, stem_out: torch.Tensor, ctx_label: torch.Tensor) -> torch.Tensor:
        """Group batch by context; run each context with its own sliced sub-network."""
        B = stem_out.size(0)
        logits = stem_out.new_zeros((B, self.fc.out_features))
        device = stem_out.device

        # Stem output is full-width; first block's input idx = full stem range.
        full_in = torch.arange(self.stage_channels[0], device=device)

        for c in range(self.num_contexts):
            sel = (ctx_label == c)
            if not sel.any():
                continue
            sub = stem_out[sel]                              # (b, C_stem, H, W)

            # Per-stage active idx for this context
            stage_idxs = [self.get_active_idx(i, c).to(device) for i in range(len(self.stages))]

            # Stage 0: input = full stem, mid/out = stage_idxs[0]
            prev_idx = full_in
            for stage_i, blocks in enumerate(self.stages):
                cur_idx = stage_idxs[stage_i]
                # First block: in = prev_idx (full for stage0; prev stage's idx otherwise)
                # but sub is currently in prev_idx channels.
                for block_j, block in enumerate(blocks):
                    in_idx = prev_idx if block_j == 0 else cur_idx
                    sub = block.forward_sliced(
                        sub, in_idx=in_idx, mid_idx=cur_idx, out_idx=cur_idx,
                        training=self.training,
                    )
                # After this stage, sub is in cur_idx channels
                prev_idx = cur_idx

            # Pool + sliced fc
            feat = self.pool(sub).flatten(1)                 # (b, k_last)
            w = self.fc.weight.index_select(1, prev_idx)     # (num_classes, k_last)
            sub_logits = F.linear(feat, w, self.fc.bias)
            logits[sel] = sub_logits

        return logits

    # ---- analysis ----

    @torch.no_grad()
    def mask_iou_matrix(self, stage_i: int) -> torch.Tensor:
        """Pairwise IoU of active sets across contexts at stage_i. Requires frozen."""
        assert self._frozen, "freeze_masks() first"
        idx = getattr(self, f"frozen_idx_{stage_i}")         # (num_ctx, k)
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
