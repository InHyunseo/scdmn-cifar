"""
Main experiment runner.

Runs three models on the same MultiContextCIFAR benchmark and produces:
- Test accuracy per context (Table)
- Gate trajectory per epoch (SCDMN)
- Linear probe + mask IoU analysis (SCDMN)

Usage (from /home/claude/scdmn):
    python -m experiments.run_all --epochs 30 --mode cnn

For a fast smoke test:
    python -m experiments.run_all --epochs 2 --train_size 2000 --quick
"""
import argparse
import json
import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data.multi_context_cifar import MultiContextCIFAR
from experiments.trainer import TrainConfig, train, evaluate, build_loaders, model_forward
from utils.analysis import run_full_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--train_size', type=int, default=10000,
                   help='Samples per context for training')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--severity', type=int, default=3)
    p.add_argument('--mode', type=str, default='cnn', choices=['cnn', 'oracle'],
                   help='Context mode for SCDMN')
    p.add_argument('--sparsity', type=float, default=0.5)
    p.add_argument('--indep_width_mult', type=float, default=1.0,
                   help='Per-expert width multiplier for independent. '
                        '1.0 = full (4x params upper bound). '
                        '0.5 = matches SCDMN active-channel budget (fair capacity comparison).')
    p.add_argument('--data_root', type=str, default='./data_cache')
    p.add_argument('--save_dir', type=str, default='./experiments/runs')
    p.add_argument('--skip_baseline', action='store_true')
    p.add_argument('--skip_independent', action='store_true')
    p.add_argument('--skip_scdmn', action='store_true')
    p.add_argument('--run_sliced', action='store_true',
                   help='Also run SCDMN-Sliced (oracle, true channel slicing).')
    p.add_argument('--freeze_epoch', type=int, default=5,
                   help='Epoch at which SCDMN-Sliced freezes its top-k masks.')
    p.add_argument('--quick', action='store_true', help='Quick smoke test (overrides several args)')
    p.add_argument('--no_official_c', action='store_true',
                   help='Force use of synthetic corruptions instead of CIFAR-10-C files')
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.epochs = 2
        args.train_size = 1000
        args.batch_size = 64

    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        severity=args.severity,
        sparsity=args.sparsity,
        indep_width_mult=args.indep_width_mult,
        use_official_c=(not args.no_official_c),
        train_size_per_context=args.train_size,
        data_root=args.data_root,
        save_dir=args.save_dir,
    )

    summary = {}

    # 1) Baseline: single ResNet18 on all contexts mixed
    if not args.skip_baseline:
        cfg = TrainConfig(model_type='baseline', run_name='baseline', **common_kwargs)
        _, hist = train(cfg)
        summary['baseline'] = hist[-1]

    # 2) Independent experts: one ResNet18 per context, oracle routing
    if not args.skip_independent:
        cfg = TrainConfig(model_type='independent', run_name='independent', **common_kwargs)
        _, hist = train(cfg)
        summary['independent'] = hist[-1]

    # 3) SCDMN
    if not args.skip_scdmn:
        cfg = TrainConfig(model_type='scdmn', context_mode=args.mode,
                          run_name=f'scdmn_{args.mode}', **common_kwargs)
        model, hist = train(cfg)
        summary[f'scdmn_{args.mode}'] = hist[-1]

        # Run analysis on the final SCDMN model
        _, test_loader = build_loaders(cfg)
        print("\n[Analysis] Running linear probe + mask IoU on SCDMN...")
        analysis = run_full_analysis(
            model=model,
            test_loader=test_loader,
            device=cfg.device,
            context_mode=args.mode,
            save_dir=cfg.save_dir,
            run_name=cfg.run_name,
        )
        print(f"[Analysis] Gates: {analysis['gates']}")
        print(f"[Analysis] Context probe per stage: {analysis['probe_context_per_stage']}")
        print(f"[Analysis] Class   probe per stage: {analysis['probe_class_per_stage']}")

    # 4) SCDMN-Sliced (oracle, true channel slicing)
    if args.run_sliced:
        cfg = TrainConfig(
            model_type='scdmn_sliced',
            context_mode='oracle',
            run_name=f'scdmn_sliced_s{args.sparsity}',
            mask_freeze_epoch=args.freeze_epoch,
            **common_kwargs,
        )
        _, hist = train(cfg)
        summary[f'scdmn_sliced_s{args.sparsity}'] = hist[-1]

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for name, result in summary.items():
        per = result['per_context']
        print(f"{name:20s}  overall={result['test_overall']:.4f}   " +
              "  ".join(f"{k}={v:.3f}" for k, v in per.items()))
        if result.get('gates') is not None:
            print(f"{'':20s}  gates={[f'{g:.3f}' for g in result['gates']]}")

    with open(Path(args.save_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    main()