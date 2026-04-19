"""
Unified trainer for the three model variants:
    - 'baseline'    : single ResNet18 trained on mixed contexts
    - 'independent' : one ResNet18 per context, oracle routing
    - 'scdmn'       : proposed Scene-Conditional Dynamic Mask Network

All three share the same data loader (MultiContextCIFAR) and the same
per-context evaluation, so numbers are directly comparable.
"""
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.multi_context_cifar import MultiContextCIFAR, CONTEXT_NAMES, NUM_CONTEXTS
from models import SCDMNResNet18, ResNet18CIFAR, IndependentExperts, SCDMNSliced


@dataclass
class TrainConfig:
    model_type: str = 'scdmn'              # 'baseline' | 'independent' | 'scdmn'
    context_mode: str = 'cnn'              # for scdmn: 'cnn' | 'oracle'
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    sparsity: float = 0.5
    indep_width_mult: float = 1.0          # independent: per-expert width multiplier
    z_dim: int = 128
    severity: int = 3
    use_official_c: bool = True
    train_size_per_context: Optional[int] = 10000  # keep run short
    warmup_soft_mask_epochs: int = 2       # SCDMN: use soft sigmoid mask for warmup
    mask_freeze_epoch: int = 5             # SCDMN-Sliced: epoch at which top-k masks are frozen
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    data_root: str = './data_cache'
    save_dir: str = './experiments/runs'
    run_name: str = 'run'
    log_every: int = 50


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_type == 'baseline':
        return ResNet18CIFAR(num_classes=10)
    if cfg.model_type == 'independent':
        return IndependentExperts(
            num_classes=10,
            num_contexts=NUM_CONTEXTS,
            width_mult=cfg.indep_width_mult,
        )
    if cfg.model_type == 'scdmn':
        return SCDMNResNet18(
            num_classes=10,
            num_contexts=NUM_CONTEXTS,
            context_mode=cfg.context_mode,
            z_dim=cfg.z_dim,
            sparsity=cfg.sparsity,
        )
    if cfg.model_type == 'scdmn_sliced':
        return SCDMNSliced(
            num_classes=10,
            num_contexts=NUM_CONTEXTS,
            sparsity=cfg.sparsity,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def model_forward(model: nn.Module, x: torch.Tensor, ctx: torch.Tensor, cfg: TrainConfig, epoch: int) -> torch.Tensor:
    """Unified forward signature across model types."""
    if cfg.model_type == 'scdmn':
        hard = epoch >= cfg.warmup_soft_mask_epochs
        if cfg.context_mode == 'oracle':
            return model(x, ctx_label=ctx, hard_mask=hard)
        return model(x, hard_mask=hard)
    if cfg.model_type == 'scdmn_sliced':
        return model(x, ctx_label=ctx)
    if cfg.model_type == 'independent':
        return model(x, ctx_label=ctx)
    return model(x)


def build_loaders(cfg: TrainConfig):
    train_ds = MultiContextCIFAR(
        data_root=cfg.data_root,
        train=True,
        use_official_c=cfg.use_official_c,
        severity=cfg.severity,
        train_size_per_context=cfg.train_size_per_context,
        augment=True,
    )
    test_ds = MultiContextCIFAR(
        data_root=cfg.data_root,
        train=False,
        use_official_c=cfg.use_official_c,
        severity=cfg.severity,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=(cfg.device == 'cuda'), drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=(cfg.device == 'cuda'),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig, epoch: int):
    """Returns dict: {'overall': float, 'per_context': {name: acc}}"""
    model.eval()
    total = {c: 0 for c in range(NUM_CONTEXTS)}
    correct = {c: 0 for c in range(NUM_CONTEXTS)}

    for x, y, ctx in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        ctx = ctx.to(cfg.device, non_blocking=True)

        logits = model_forward(model, x, ctx, cfg, epoch)
        pred = logits.argmax(dim=1)

        for c in range(NUM_CONTEXTS):
            m = (ctx == c)
            if m.any():
                total[c] += int(m.sum().item())
                correct[c] += int((pred[m] == y[m]).sum().item())

    per_context = {}
    for c in range(NUM_CONTEXTS):
        if total[c] == 0:
            per_context[CONTEXT_NAMES[c]] = float('nan')
        else:
            per_context[CONTEXT_NAMES[c]] = correct[c] / total[c]

    overall_total = sum(total.values())
    overall_correct = sum(correct.values())
    overall = overall_correct / overall_total if overall_total > 0 else float('nan')

    return {'overall': overall, 'per_context': per_context}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.save_dir) / f"{cfg.run_name}.log"

    def log(msg: str):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + "\n")

    log(f"=== Run: {cfg.run_name} ===")
    log(f"Config: {cfg}")

    train_loader, test_loader = build_loaders(cfg)
    log(f"Train size: {len(train_loader.dataset)}   Test size: {len(test_loader.dataset)}")

    model = build_model(cfg).to(cfg.device)
    log(f"Model: {cfg.model_type}   Params: {count_params(model):,}")

    if cfg.model_type == 'scdmn_sliced':
        score_params = list(model.channel_scores.parameters())
        score_ids = {id(p) for p in score_params}
        other_params = [p for p in model.parameters() if id(p) not in score_ids]
        optim = torch.optim.SGD(
            [
                {'params': other_params, 'lr': cfg.lr},
                {'params': score_params, 'lr': cfg.lr * 10.0, 'weight_decay': 0.0},
            ],
            lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )
    else:
        optim = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    best_overall = 0.0
    history = []

    for epoch in range(cfg.epochs):
        # SCDMN-Sliced: freeze top-k masks at the configured epoch
        if cfg.model_type == 'scdmn_sliced' and epoch == cfg.mask_freeze_epoch and not model.is_frozen():
            model.freeze_masks()
            log(f"[scdmn_sliced] Froze masks at epoch {epoch}. Switching to sliced forward.")

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (x, y, ctx) in enumerate(train_loader):
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            ctx = ctx.to(cfg.device, non_blocking=True)

            logits = model_forward(model, x, ctx, cfg, epoch)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += x.size(0)

            if step % cfg.log_every == 0:
                log(f"  epoch {epoch:03d} step {step:04d}   loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        eval_result = evaluate(model, test_loader, cfg, epoch)
        dt = time.time() - t0

        # Log gate values for SCDMN
        extra = ""
        if cfg.model_type == 'scdmn':
            gates = model.get_gate_values()
            extra = f"   gates={[f'{g:.3f}' for g in gates]}"
        elif cfg.model_type == 'scdmn_sliced':
            extra = f"   frozen={model.is_frozen()}"

        log(f"Epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_overall={eval_result['overall']:.4f}  "
            f"per_ctx={ {k: f'{v:.3f}' for k, v in eval_result['per_context'].items()} }  "
            f"time={dt:.1f}s{extra}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_overall': eval_result['overall'],
            'per_context': eval_result['per_context'],
            'gates': model.get_gate_values() if cfg.model_type == 'scdmn' else None,
        })

        if eval_result['overall'] > best_overall:
            best_overall = eval_result['overall']
            ckpt = Path(cfg.save_dir) / f"{cfg.run_name}_best.pt"
            torch.save({
                'model_state': model.state_dict(),
                'cfg': cfg.__dict__,
                'epoch': epoch,
                'eval': eval_result,
            }, ckpt)

    # Save final and history
    final_ckpt = Path(cfg.save_dir) / f"{cfg.run_name}_final.pt"
    torch.save({
        'model_state': model.state_dict(),
        'cfg': cfg.__dict__,
        'history': history,
    }, final_ckpt)
    log(f"Best overall: {best_overall:.4f}")
    log(f"Saved: {final_ckpt}")
    return model, history