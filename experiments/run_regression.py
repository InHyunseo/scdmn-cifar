"""
Regression experiment runner: Single (Mixed) vs SCDMN-Sliced.

Usage:
  python -m experiments.run_regression --epochs 30
  python -m experiments.run_regression --quick --no_official_c
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.trainer_reg import TrainConfigReg, train, evaluate, build_loaders
from data.multi_context_cifar_reg import CONTEXT_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--train_size', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--severity', type=int, default=3)
    p.add_argument('--sparsity', type=float, default=0.5)
    p.add_argument('--freeze_epoch', type=int, default=5)
    p.add_argument('--data_root', type=str, default='./data_cache')
    p.add_argument('--save_dir', type=str, default='./experiments/runs/regression')
    p.add_argument('--skip_single', action='store_true')
    p.add_argument('--skip_sliced', action='store_true')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no_official_c', action='store_true')
    return p.parse_args()


def ensure_pseudo_targets(data_root: str):
    d = Path(data_root)
    if (d / 'pseudo_train.npy').exists() and (d / 'pseudo_test.npy').exists():
        return
    print(f"[run_regression] Pseudo target files missing under {d}. Generating...")
    from scripts.make_pseudo_targets import main as make_main
    sys.argv = ['make_pseudo_targets', '--data_root', str(d)]
    make_main()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.train_size = 1000
        args.batch_size = 64
        args.freeze_epoch = 1

    ensure_pseudo_targets(args.data_root)

    common = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        severity=args.severity,
        sparsity=args.sparsity,
        use_official_c=(not args.no_official_c),
        train_size_per_context=args.train_size,
        mask_freeze_epoch=args.freeze_epoch,
        data_root=args.data_root,
        save_dir=args.save_dir,
    )

    results = {}

    if not args.skip_single:
        cfg = TrainConfigReg(
            model_type='single_reg',
            run_name='single_reg',
            **common,
        )
        _, hist = train(cfg)
        results['single_reg'] = hist[-1]

    if not args.skip_sliced:
        cfg = TrainConfigReg(
            model_type='scdmn_sliced_reg',
            run_name=f'sliced_reg_s{args.sparsity}',
            **common,
        )
        _, hist = train(cfg)
        results['scdmn_sliced_reg'] = hist[-1]

    # Summary table
    print("\n=== Final per-context MAE ===")
    header = f"{'context':<18}" + "".join(f"{m:>16}" for m in results.keys())
    if 'single_reg' in results and 'scdmn_sliced_reg' in results:
        header += f"{'delta(S-SCDMN)':>18}"
    print(header)
    for ctx in CONTEXT_NAMES:
        row = f"{ctx:<18}"
        for m in results.keys():
            v = results[m]['per_context'].get(ctx, float('nan'))
            row += f"{v:>16.4f}"
        if 'single_reg' in results and 'scdmn_sliced_reg' in results:
            d = results['single_reg']['per_context'][ctx] - results['scdmn_sliced_reg']['per_context'][ctx]
            row += f"{d:>+18.4f}"
        print(row)
    row = f"{'OVERALL':<18}"
    for m in results.keys():
        row += f"{results[m]['test_overall_mae']:>16.4f}"
    if 'single_reg' in results and 'scdmn_sliced_reg' in results:
        d = results['single_reg']['test_overall_mae'] - results['scdmn_sliced_reg']['test_overall_mae']
        row += f"{d:>+18.4f}"
    print(row)

    out = Path(args.save_dir) / 'summary_reg.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {out}")


if __name__ == '__main__':
    main()
