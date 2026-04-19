"""
Analysis & figures for SCDMN-Sliced.

Produces a single multi-panel PNG with:
  (1) Per-stage IoU heatmaps (4 panels, one per ResNet stage)
  (2) Per-context accuracy bars (sliced vs independent vs baseline if available)
  (3) Scatter: per-context mean IoU-with-others vs accuracy gap to independent

Usage:
    python -m utils.analyze_sliced \
        --sliced_ckpt experiments/runs/scdmn_sliced_s0.5_final.pt \
        --indep_ckpt  experiments/runs/independent_final.pt \
        --baseline_ckpt experiments/runs/baseline_final.pt \
        --out experiments/runs/sliced_analysis.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.multi_context_cifar import CONTEXT_NAMES
from models import SCDMNSliced


def load_per_ctx_acc(ckpt_path: str) -> dict | None:
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # _final.pt has 'history'; _best.pt has 'eval'
    if 'eval' in ck:
        return ck['eval']['per_context']
    if 'history' in ck and ck['history']:
        return ck['history'][-1]['per_context']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sliced_ckpt', required=True)
    ap.add_argument('--indep_ckpt', default='')
    ap.add_argument('--baseline_ckpt', default='')
    ap.add_argument('--sparsity', type=float, default=0.5)
    ap.add_argument('--out', default='experiments/runs/sliced_analysis.png')
    args = ap.parse_args()

    # Load sliced model
    ck = torch.load(args.sliced_ckpt, map_location='cpu', weights_only=False)
    m = SCDMNSliced(num_classes=10, num_contexts=len(CONTEXT_NAMES), sparsity=args.sparsity)
    m.load_state_dict(ck['model_state'])
    m.freeze_masks()  # buffer already populated; safe to recompute

    # Per-context accuracy
    sliced_acc = None
    if 'history' in ck and ck['history']:
        sliced_acc = ck['history'][-1]['per_context']
    indep_acc = load_per_ctx_acc(args.indep_ckpt)
    base_acc = load_per_ctx_acc(args.baseline_ckpt)

    # IoU matrices
    stage_channels = m.stage_channels
    ious = [m.mask_iou_matrix(i).numpy() for i in range(len(stage_channels))]

    # Random IoU baseline for sparsity s with N channels: roughly s on large N
    random_iou = args.sparsity / (2 - args.sparsity)  # = |A∩B|/|A∪B| in expectation
    n_ctx = len(CONTEXT_NAMES)

    # Figure: 2 rows x 4 cols. Top row = 4 IoU heatmaps. Bottom: bars + scatter (span 4 cols as 2+2).
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.5, wspace=0.4)

    # --- Top: IoU heatmaps ---
    for i, (iou, c) in enumerate(zip(ious, stage_channels)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(iou, vmin=0.0, vmax=1.0, cmap='Reds')
        ax.set_title(f'Stage {i} (C={c}) IoU\nrandom~{random_iou:.2f}', fontsize=10)
        ax.set_xticks(range(n_ctx))
        ax.set_yticks(range(n_ctx))
        ax.set_xticklabels(CONTEXT_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(CONTEXT_NAMES, fontsize=8)
        for r in range(n_ctx):
            for col in range(n_ctx):
                txt_color = 'white' if iou[r, col] > 0.6 else 'black'
                ax.text(col, r, f'{iou[r, col]:.2f}', ha='center', va='center',
                        color=txt_color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Bottom-left: per-context accuracy bars ---
    ax_bar = fig.add_subplot(gs[1, 0:2])
    x = np.arange(n_ctx)
    width = 0.27
    series = []
    if sliced_acc is not None:
        series.append(('SCDMN-Sliced', sliced_acc, '#d62728'))
    if indep_acc is not None:
        series.append(('Independent', indep_acc, '#2ca02c'))
    if base_acc is not None:
        series.append(('Baseline', base_acc, '#1f77b4'))

    for k, (label, accs, color) in enumerate(series):
        vals = [accs.get(name, np.nan) for name in CONTEXT_NAMES]
        offset = (k - (len(series) - 1) / 2) * width
        ax_bar.bar(x + offset, vals, width, label=label, color=color)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CONTEXT_NAMES, rotation=20)
    ax_bar.set_ylabel('Test accuracy')
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_title('Per-context accuracy')
    ax_bar.legend(loc='lower left')
    ax_bar.grid(axis='y', alpha=0.3)

    # --- Bottom-right: scatter IoU vs gap ---
    ax_sc = fig.add_subplot(gs[1, 2:4])
    if sliced_acc is not None and indep_acc is not None:
        # Per-context mean IoU with OTHER contexts, averaged across stages
        mean_iou_per_ctx = []
        for c in range(n_ctx):
            vals = []
            for stage_iou in ious:
                row = np.delete(stage_iou[c], c)
                vals.extend(row.tolist())
            mean_iou_per_ctx.append(float(np.mean(vals)))
        gaps = [indep_acc[name] - sliced_acc[name] for name in CONTEXT_NAMES]

        ax_sc.scatter(mean_iou_per_ctx, gaps, s=120, c='#d62728', zorder=3)
        for i, name in enumerate(CONTEXT_NAMES):
            ax_sc.annotate(name, (mean_iou_per_ctx[i], gaps[i]),
                           xytext=(6, 6), textcoords='offset points', fontsize=10)
        ax_sc.axhline(0, color='gray', lw=0.8, alpha=0.6)
        ax_sc.axvline(random_iou, color='gray', lw=0.8, ls='--', alpha=0.6,
                      label=f'random IoU ~{random_iou:.2f}')
        ax_sc.set_xlabel('Mean IoU with other contexts (avg over stages)')
        ax_sc.set_ylabel('Accuracy gap: independent − sliced')
        ax_sc.set_title('Channel sharing vs accuracy cost')
        ax_sc.legend(loc='best')
        ax_sc.grid(alpha=0.3)
    else:
        ax_sc.text(0.5, 0.5, 'Need both sliced and independent ckpts',
                   ha='center', va='center', transform=ax_sc.transAxes)
        ax_sc.set_axis_off()

    fig.suptitle(f'SCDMN-Sliced analysis (sparsity={args.sparsity})', fontsize=13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Saved: {args.out}')

    # Also dump numbers as text
    print('\n--- IoU per stage ---')
    for i, iou in enumerate(ious):
        print(f'Stage {i}:'); print(np.round(iou, 3))
    if sliced_acc is not None:
        print('\nSliced per-ctx:', {k: f'{v:.3f}' for k, v in sliced_acc.items()})
    if indep_acc is not None:
        print('Indep  per-ctx:', {k: f'{v:.3f}' for k, v in indep_acc.items()})


if __name__ == '__main__':
    main()
