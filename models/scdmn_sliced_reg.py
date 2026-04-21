"""
Regression variant of SCDMN-Sliced.

Identical architecture to SCDMNSliced except:
  - fc out_features = 1
  - output passes through tanh (both soft and sliced paths)
  - forward returns shape (B, 1); caller should squeeze(-1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scdmn_sliced import SlicedBasicBlock


class SCDMNSlicedReg(nn.Module):
    def __init__(
        self,
        num_contexts: int = 4,
        sparsity: float = 0.5,
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
        self.fc = nn.Linear(self.stage_channels[-1], 1)

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

    def forward(self, x: torch.Tensor, ctx_label: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        B = out.size(0)
        device = out.device
        preds = out.new_zeros((B, 1))

        full_in = torch.arange(self.stage_channels[0], device=device)

        for c in range(self.num_contexts):
            sel = (ctx_label == c)
            if not sel.any():
                continue
            sub = out[sel]

            if self._frozen:
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
                sub_out = F.linear(feat, w, self.fc.bias)
            else:
                m_prev = torch.ones(self.stage_channels[0], device=device, dtype=sub.dtype)
                for stage_i, blocks in enumerate(self.stages):
                    m_cur = torch.sigmoid(self.channel_scores[stage_i][c])
                    for block_j, block in enumerate(blocks):
                        m_in = m_prev if block_j == 0 else m_cur
                        sub = block.forward_soft(
                            sub, m_in=m_in, m_mid=m_cur, m_out=m_cur,
                            training=self.training,
                        )
                    m_prev = m_cur
                feat = self.pool(sub).flatten(1)
                fc_w = self.fc.weight * m_prev.view(1, -1)
                sub_out = F.linear(feat, fc_w, self.fc.bias)

            preds[sel] = torch.tanh(sub_out)

        return preds

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
