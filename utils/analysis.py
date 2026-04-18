"""
Analysis utilities for SCDMN.

Two core analyses to validate the hypothesis:
1. Gate distribution per layer: are gates low at shallow layers, high at deep?
2. Linear Probe: at which depth do features become context-separable?

If the two curves agree (shared early, context-specific late), we claim
the mask structure is learning the expected hierarchy.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.multi_context_cifar import CONTEXT_NAMES, NUM_CONTEXTS


@torch.no_grad()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    context_mode: str = 'cnn',
    max_samples: int = 2000,
) -> Dict:
    """
    Runs the model in feature-return mode and collects stage-level pooled features
    along with class labels and context labels.

    Returns:
        {
          'features': list[np.ndarray]   # one (N, C_i) array per stage
          'labels':   np.ndarray (N,)    class labels
          'contexts': np.ndarray (N,)    context labels
          'gates':    list[float]        current gate values
          'masks':    list[np.ndarray]   (N, C_i) masks per stage
        }
    """
    model.eval()
    feats_per_stage = None
    masks_per_stage = None
    all_labels = []
    all_contexts = []
    collected = 0

    for x, y, ctx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)

        if context_mode == 'oracle':
            logits, feats = model(x, ctx_label=ctx, return_features=True, hard_mask=True)
        else:
            logits, feats = model(x, return_features=True, hard_mask=True)

        # Global average pool each stage feature to (B, C_i)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1).cpu().numpy() for f in feats]
        masks = [m.cpu().numpy() for m in model.get_last_masks()]

        if feats_per_stage is None:
            feats_per_stage = [[] for _ in pooled]
            masks_per_stage = [[] for _ in masks]

        for i, p in enumerate(pooled):
            feats_per_stage[i].append(p)
        for i, m in enumerate(masks):
            masks_per_stage[i].append(m)

        all_labels.append(y.cpu().numpy())
        all_contexts.append(ctx.cpu().numpy())
        collected += x.size(0)
        if collected >= max_samples:
            break

    out = {
        'features': [np.concatenate(fs, axis=0)[:max_samples] for fs in feats_per_stage],
        'masks':    [np.concatenate(ms, axis=0)[:max_samples] for ms in masks_per_stage],
        'labels':   np.concatenate(all_labels, axis=0)[:max_samples],
        'contexts': np.concatenate(all_contexts, axis=0)[:max_samples],
        'gates':    model.get_gate_values(),
    }
    return out


def linear_probe_context_accuracy(features: np.ndarray, contexts: np.ndarray, num_contexts: int) -> float:
    """
    Fits a multinomial logistic regression from `features` to `contexts`.
    Returns held-out accuracy (50/50 split).

    Interpretation:
      - High accuracy  -> features carry strong context signal (context-specific)
      - Chance-level   -> features are context-invariant (shared representation)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    N = features.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    split = N // 2
    tr_idx, te_idx = perm[:split], perm[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(features[tr_idx])
    X_te = scaler.transform(features[te_idx])
    y_tr = contexts[tr_idx]
    y_te = contexts[te_idx]

    clf = LogisticRegression(max_iter=1000, multi_class='auto', n_jobs=1)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def linear_probe_class_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Same as above but predicting class label. Sanity check: should increase with depth."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    N = features.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    split = N // 2
    tr_idx, te_idx = perm[:split], perm[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(features[tr_idx])
    X_te = scaler.transform(features[te_idx])

    clf = LogisticRegression(max_iter=1000, multi_class='auto', n_jobs=1)
    clf.fit(X_tr, labels[tr_idx])
    return float(clf.score(X_te, labels[te_idx]))


def mask_overlap_iou(masks: np.ndarray, contexts: np.ndarray, num_contexts: int) -> np.ndarray:
    """
    For a single layer's mask array (N, C), compute the per-context MEAN mask
    (thresholded at 0.5 after averaging) and return pairwise IoU matrix (num_contexts, num_contexts).

    Low IoU between different contexts => mask generator is routing to different channels.
    """
    per_ctx_mask = []
    for c in range(num_contexts):
        m = masks[contexts == c]
        if len(m) == 0:
            per_ctx_mask.append(np.zeros(masks.shape[1], dtype=bool))
            continue
        mean = (m > 0).mean(axis=0)  # fraction of samples that kept each channel
        per_ctx_mask.append(mean > 0.5)

    iou = np.zeros((num_contexts, num_contexts))
    for i in range(num_contexts):
        for j in range(num_contexts):
            a, b = per_ctx_mask[i], per_ctx_mask[j]
            union = np.logical_or(a, b).sum()
            inter = np.logical_and(a, b).sum()
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def run_full_analysis(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    context_mode: str,
    save_dir: str,
    run_name: str,
):
    """
    Runs all analyses and saves results (JSON + a matplotlib figure if possible).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_features(model, test_loader, device, context_mode=context_mode)

    results = {
        'gates': collected['gates'],
        'num_stages': len(collected['features']),
        'probe_context_per_stage': [],
        'probe_class_per_stage': [],
        'mask_iou_per_stage': [],
    }

    for i, feat in enumerate(collected['features']):
        ctx_acc = linear_probe_context_accuracy(feat, collected['contexts'], NUM_CONTEXTS)
        cls_acc = linear_probe_class_accuracy(feat, collected['labels'])
        iou = mask_overlap_iou(collected['masks'][i], collected['contexts'], NUM_CONTEXTS)
        results['probe_context_per_stage'].append(ctx_acc)
        results['probe_class_per_stage'].append(cls_acc)
        results['mask_iou_per_stage'].append(iou.tolist())

    # Save JSON
    with open(save_dir / f"{run_name}_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        stages = np.arange(1, results['num_stages'] + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # (1) Gates per stage
        axes[0].bar(stages, results['gates'], color='steelblue')
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel('Stage')
        axes[0].set_ylabel('Gate g_i')
        axes[0].set_title('Gate values per stage')

        # (2) Linear probe accuracy
        axes[1].plot(stages, results['probe_context_per_stage'], 'o-', label='context', color='crimson')
        axes[1].plot(stages, results['probe_class_per_stage'], 's-', label='class', color='seagreen')
        axes[1].axhline(1.0 / NUM_CONTEXTS, ls='--', color='grey', label=f'chance (ctx)={1/NUM_CONTEXTS:.2f}')
        axes[1].axhline(0.1, ls=':', color='grey', label='chance (cls)=0.10')
        axes[1].set_xlabel('Stage')
        axes[1].set_ylabel('Linear probe accuracy')
        axes[1].set_title('Probe: how context/class-separable are features?')
        axes[1].legend()
        axes[1].set_ylim(0, 1.05)

        # (3) Mask IoU at deepest stage (heatmap)
        last_iou = np.array(results['mask_iou_per_stage'][-1])
        im = axes[2].imshow(last_iou, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_xticks(range(NUM_CONTEXTS))
        axes[2].set_yticks(range(NUM_CONTEXTS))
        axes[2].set_xticklabels(CONTEXT_NAMES, rotation=30)
        axes[2].set_yticklabels(CONTEXT_NAMES)
        axes[2].set_title(f'Mask IoU at stage {results["num_stages"]}\n(low = contexts route differently)')
        plt.colorbar(im, ax=axes[2])

        plt.suptitle(f'SCDMN analysis — {run_name}')
        plt.tight_layout()
        plt.savefig(save_dir / f"{run_name}_analysis.png", dpi=130)
        plt.close()
    except ImportError:
        pass

    return results