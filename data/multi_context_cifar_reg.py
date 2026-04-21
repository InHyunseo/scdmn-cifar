"""
Regression variant of MultiContextCIFAR.

Each sample returns (image_tensor, pseudo_target_float, context_label)
instead of a class label. Pseudo targets are keyed by the CIFAR source
index (the same across all 4 contexts), which creates mode-averaging
pressure on a single mixed model.

Pseudo target files are produced by scripts/make_pseudo_targets.py.
"""
from pathlib import Path

import numpy as np
import torch

from data.multi_context_cifar import (
    MultiContextCIFAR,
    CONTEXT_NAMES,
    NUM_CONTEXTS,
    _to_tensor_normalize,
)


class MultiContextCIFARReg(MultiContextCIFAR):
    def __init__(self, *args, pseudo_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pseudo_path is None:
            name = 'pseudo_train.npy' if self.train else 'pseudo_test.npy'
            pseudo_path = Path(self.data_root) / name
        pseudo_path = Path(pseudo_path)
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"Pseudo target file not found: {pseudo_path}. "
                f"Run: python -m scripts.make_pseudo_targets --data_root {self.data_root}"
            )
        self.pseudo = np.load(pseudo_path).astype(np.float32)
        expected = len(self.cifar_data)
        if len(self.pseudo) != expected:
            raise ValueError(
                f"Pseudo target length {len(self.pseudo)} != CIFAR split size {expected}"
            )

    def __getitem__(self, idx):
        ctx_id, src_idx = self.index[idx]
        img_uint8, _ = self._get_raw(ctx_id, src_idx)

        if self.augment:
            if np.random.rand() < 0.5:
                img_uint8 = img_uint8[:, ::-1, :].copy()
            padded = np.pad(img_uint8, ((4, 4), (4, 4), (0, 0)), mode='reflect')
            y0 = np.random.randint(0, 9)
            x0 = np.random.randint(0, 9)
            img_uint8 = padded[y0:y0 + 32, x0:x0 + 32, :]

        img_tensor = _to_tensor_normalize(img_uint8)
        target = torch.tensor(float(self.pseudo[src_idx]), dtype=torch.float32)
        return img_tensor, target, ctx_id


__all__ = ["MultiContextCIFARReg", "CONTEXT_NAMES", "NUM_CONTEXTS"]
