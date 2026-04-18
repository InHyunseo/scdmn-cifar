"""
Multi-context CIFAR dataset.

Contexts:
    0: clean (CIFAR-10 original)
    1: brightness  (CIFAR-10-C, severity configurable)
    2: frost       (CIFAR-10-C)
    3: gaussian_noise (CIFAR-10-C)

CIFAR-10-C format:
    Download from https://zenodo.org/records/2535967
    Each corruption is a (50000, 32, 32, 3) uint8 .npy array (5 severities x 10000 test imgs).
    Labels: labels.npy, shape (50000,) — same labels.npy for every corruption.

Expected layout:
    data_root/
        cifar-10-batches-py/...         (torchvision's standard CIFAR-10)
        CIFAR-10-C/
            brightness.npy
            frost.npy
            gaussian_noise.npy
            labels.npy

For training, we also provide a fallback 'synthetic' mode that applies the
corruptions ourselves to plain CIFAR-10, so the pipeline still works if the
CIFAR-10-C files are not available.
"""
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Canonical order of contexts. Index = context label.
CONTEXT_NAMES = ['clean', 'brightness', 'frost', 'gaussian_noise']
NUM_CONTEXTS = len(CONTEXT_NAMES)

# CIFAR-10 statistics (used for normalization)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _to_tensor_normalize(img_uint8: np.ndarray) -> torch.Tensor:
    """img_uint8: HxWx3 uint8 -> 3xHxW float tensor, normalized."""
    img = torch.from_numpy(img_uint8).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (img - mean) / std


def _apply_synthetic_corruption(img_uint8: np.ndarray, context_id: int, severity: int = 3) -> np.ndarray:
    """
    Fallback: apply a corruption in-process so we don't need CIFAR-10-C files.
    These are simplified versions; they match the SPIRIT of brightness/frost/noise
    but are NOT a substitute for the official CIFAR-10-C benchmark.
    """
    img = img_uint8.astype(np.float32) / 255.0
    rng = np.random.RandomState(seed=None)

    if context_id == 0:
        out = img
    elif context_id == 1:  # brightness
        scale = 1.0 + 0.1 * severity  # severity 3 -> 1.3x
        out = np.clip(img * scale, 0, 1)
    elif context_id == 2:  # 'frost' approximated as low-contrast haze + blue tint
        haze = np.ones_like(img) * 0.85
        alpha = 0.15 * severity  # severity 3 -> 0.45
        out = (1 - alpha) * img + alpha * haze
        out[..., 2] = np.clip(out[..., 2] + 0.05 * severity, 0, 1)  # blue tint
    elif context_id == 3:  # gaussian noise
        sigma = 0.04 * severity  # severity 3 -> 0.12
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        out = np.clip(img + noise, 0, 1)
    else:
        out = img

    return (out * 255).astype(np.uint8)


class MultiContextCIFAR(Dataset):
    """
    Combines CIFAR-10 (clean) with three CIFAR-10-C corruptions into a single
    dataset. Each sample is (image_tensor, class_label, context_label).

    Args:
        data_root: directory containing cifar-10-batches-py and optionally CIFAR-10-C
        train: if True, use CIFAR-10 train split for the clean context.
               Corruption files are test-set-derived; in 'train' mode we slice them
               to the first N_train images (with replacement from severities).
        contexts: which context ids to include (default: all 4)
        use_official_c: if True and files exist, load CIFAR-10-C .npy files;
                        otherwise fall back to synthetic.
        severity: severity for CIFAR-10-C (1..5). Each severity block is 10000 images.
        train_size_per_context: if train=True, how many images per context.
                                 None -> use all available (50000 for clean).
    """
    def __init__(
        self,
        data_root: str,
        train: bool = True,
        contexts: Optional[List[int]] = None,
        use_official_c: bool = True,
        severity: int = 3,
        train_size_per_context: Optional[int] = None,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.train = train
        self.severity = severity
        self.augment = augment and train
        self.contexts = contexts if contexts is not None else list(range(NUM_CONTEXTS))

        # Always need CIFAR-10 itself (for clean context and for labels)
        cifar = datasets.CIFAR10(
            root=str(self.data_root),
            train=train,
            download=True,
            transform=None,  # we handle transforms manually
        )
        # Stack into numpy arrays for uniform handling
        self.cifar_data = np.stack([np.array(img) for img, _ in cifar])   # (N, 32, 32, 3)
        self.cifar_labels = np.array([label for _, label in cifar])       # (N,)

        N = len(self.cifar_data)

        # Check for CIFAR-10-C availability
        c_dir = self.data_root / 'CIFAR-10-C'
        has_c_files = use_official_c and all(
            (c_dir / f'{name}.npy').exists() for name in CONTEXT_NAMES[1:]
        ) and (c_dir / 'labels.npy').exists()
        self.has_official_c = has_c_files

        if not has_c_files and use_official_c:
            print(f"[MultiContextCIFAR] CIFAR-10-C files not found at {c_dir}. "
                  f"Falling back to synthetic corruptions.")

        # Preload corruption arrays if available
        self.c_arrays = {}  # context_id -> (N_c, 32, 32, 3) uint8
        self.c_labels = None
        if has_c_files:
            self.c_labels = np.load(c_dir / 'labels.npy')  # (50000,)
            # Severity s uses block [(s-1)*10000 : s*10000]
            lo = (severity - 1) * 10000
            hi = severity * 10000
            for ctx_id, name in enumerate(CONTEXT_NAMES):
                if ctx_id == 0:
                    continue
                arr = np.load(c_dir / f'{name}.npy')  # (50000, 32, 32, 3) uint8
                self.c_arrays[ctx_id] = arr[lo:hi]

        # Build the index: list of (context_id, source_index) tuples
        self.index = []
        for ctx_id in self.contexts:
            if ctx_id == 0:
                # Clean: use CIFAR-10 as-is
                n = N if train_size_per_context is None else min(train_size_per_context, N)
                for i in range(n):
                    self.index.append((0, i))
            else:
                if has_c_files:
                    n_c = len(self.c_arrays[ctx_id])  # 10000 per severity
                    if train:
                        # repeat to reach ~N samples if requested
                        target = train_size_per_context or N
                        for i in range(target):
                            self.index.append((ctx_id, i % n_c))
                    else:
                        # test mode: use the 10000 corruption-severity test images
                        for i in range(n_c):
                            self.index.append((ctx_id, i))
                else:
                    # Synthetic: we apply corruption on-the-fly to CIFAR-10 samples
                    n = N if train_size_per_context is None else min(train_size_per_context, N)
                    for i in range(n):
                        self.index.append((ctx_id, i))

    def __len__(self):
        return len(self.index)

    def _get_raw(self, ctx_id: int, src_idx: int):
        """Returns (img_uint8 HxWx3, class_label)."""
        if ctx_id == 0:
            return self.cifar_data[src_idx], int(self.cifar_labels[src_idx])
        if self.has_official_c:
            img = self.c_arrays[ctx_id][src_idx]
            lo = (self.severity - 1) * 10000
            label = int(self.c_labels[lo + src_idx])
            return img, label
        # Synthetic path
        base_img = self.cifar_data[src_idx]
        label = int(self.cifar_labels[src_idx])
        img = _apply_synthetic_corruption(base_img, ctx_id, severity=self.severity)
        return img, label

    def __getitem__(self, idx):
        ctx_id, src_idx = self.index[idx]
        img_uint8, class_label = self._get_raw(ctx_id, src_idx)

        # Light augmentation for training (applied on uint8 HWC)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img_uint8 = img_uint8[:, ::-1, :].copy()
            # Random crop with 4px padding
            padded = np.pad(img_uint8, ((4, 4), (4, 4), (0, 0)), mode='reflect')
            y0 = np.random.randint(0, 9)
            x0 = np.random.randint(0, 9)
            img_uint8 = padded[y0:y0 + 32, x0:x0 + 32, :]

        img_tensor = _to_tensor_normalize(img_uint8)
        return img_tensor, class_label, ctx_id