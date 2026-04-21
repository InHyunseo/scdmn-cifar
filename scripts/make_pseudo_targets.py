"""
Generate pseudo steering targets for CIFAR-10 images.

Pipeline:
  pretrained ResNet18 (ImageNet) -> 512-d features on CIFAR-10 clean
  -> PCA 1-D (fit on train, apply to train+test)
  -> z-score using train stats
  -> y = tanh(z / 2.0) in (-1, 1)

The target is keyed by the CIFAR-10 source index. At train time the
MultiContextCIFAR dataset shares the same src_idx across all 4 contexts,
so every corrupted variant of the same image gets the SAME pseudo target.
This is what creates mode-averaging pressure on a single mixed model.

Outputs:
  {data_root}/pseudo_train.npy  shape (50000,)  float32
  {data_root}/pseudo_test.npy   shape (10000,)  float32

Usage:
  python -m scripts.make_pseudo_targets --data_root ./data_cache
  python -m scripts.make_pseudo_targets --force   # re-generate
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _extract_features(images_uint8, device, batch_size=256):
    """images_uint8: (N, 32, 32, 3) uint8. Returns (N, 512) float32 features."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    N = len(images_uint8)
    feats = np.zeros((N, 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images_uint8[i:i + batch_size]
            x = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
            x = x.to(device)
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
            x = (x - mean) / std
            f = net(x)
            feats[i:i + batch_size] = f.cpu().numpy()
            if (i // batch_size) % 10 == 0:
                print(f"  feat batch {i}/{N}")
    return feats


def _pca_1d(train_feats, test_feats):
    """Fit PCA on train, project both."""
    mu = train_feats.mean(axis=0, keepdims=True)
    Xc = train_feats - mu
    # top singular direction
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0]  # (512,)
    train_proj = (train_feats - mu) @ v
    test_proj = (test_feats - mu) @ v
    return train_proj.astype(np.float32), test_proj.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data_cache')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    out_train = data_root / 'pseudo_train.npy'
    out_test = data_root / 'pseudo_test.npy'

    if out_train.exists() and out_test.exists() and not args.force:
        print(f"Pseudo targets already exist at {data_root}. Use --force to regenerate.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading CIFAR-10 train...")
    cifar_train = datasets.CIFAR10(root=str(data_root), train=True, download=True)
    train_imgs = np.stack([np.array(img) for img, _ in cifar_train])
    print(f"  train shape: {train_imgs.shape}")

    print("Loading CIFAR-10 test...")
    cifar_test = datasets.CIFAR10(root=str(data_root), train=False, download=True)
    test_imgs = np.stack([np.array(img) for img, _ in cifar_test])
    print(f"  test shape: {test_imgs.shape}")

    print("Extracting train features...")
    train_feats = _extract_features(train_imgs, device, batch_size=args.batch_size)
    print("Extracting test features...")
    test_feats = _extract_features(test_imgs, device, batch_size=args.batch_size)

    print("Fitting PCA(1) on train features...")
    train_proj, test_proj = _pca_1d(train_feats, test_feats)

    mu = train_proj.mean()
    sigma = train_proj.std() + 1e-8
    z_train = (train_proj - mu) / sigma
    z_test = (test_proj - mu) / sigma

    y_train = np.tanh(z_train / 2.0).astype(np.float32)
    y_test = np.tanh(z_test / 2.0).astype(np.float32)

    np.save(out_train, y_train)
    np.save(out_test, y_test)
    print(f"Saved: {out_train}")
    print(f"Saved: {out_test}")

    def summary(name, y):
        q = np.quantile(y, [0.0, 0.1, 0.5, 0.9, 1.0])
        print(f"  {name}: mean={y.mean():+.3f} std={y.std():.3f} "
              f"quantiles(0/10/50/90/100)={[f'{v:+.3f}' for v in q]}")

    print("Summary:")
    summary("train", y_train)
    summary("test", y_test)


if __name__ == '__main__':
    main()
