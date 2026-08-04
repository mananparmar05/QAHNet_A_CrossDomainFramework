

"""
Run full ablation study - trains all 6 variants sequentially.

Usage:
    python run_ablation.py --data_dir ./PADUFES
"""
import subprocess
import sys
import time
import json
import os


VARIANTS = {
    'A': 'CNN-only (EfficientNet-B0)',
    'B': 'CNN + Transformer',
    'C': 'B + Passive QTok',
    'D': 'B + FiLM QTok',
    'E': 'D + Metadata',
    'F': 'Full QAHNet (C1+C2+C3)',
}


def run_variant(variant, data_dir, output_dir):
    """Train a single variant."""
    print(f"\n{'#' * 60}")
    print(f"  TRAINING VARIANT {variant}: {VARIANTS[variant]}")
    print(f"{'#' * 60}\n")

    cmd = [
        sys.executable, 'train.py',
        '--variant', variant,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--epochs', '100',
        '--batch_size', '4',
        '--patience', '15',
        '--num_workers', '0',
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    print(f"\nVariant {variant} completed in {elapsed/60:.1f} minutes")
    print(f"Exit code: {result.returncode}")

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./PADUFES')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--variants', type=str, default='A,B,C,D,E,F',
                        help='Comma-separated list of variants to train')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',')]
    total_start = time.time()

    print("=" * 60)
    print("  QAHNet ABLATION STUDY")
    print(f"  Variants: {', '.join(variants)}")
    print(f"  Dataset: {args.data_dir}")
    print("=" * 60)

    results = {}
    for v in variants:
        success = run_variant(v, args.data_dir, args.output_dir)
        results[v] = 'SUCCESS' if success else 'FAILED'

    # Summary
    total_time = time.time() - total_start
    print(f"\n\n{'=' * 60}")
    print(f"  ABLATION STUDY COMPLETE ({total_time/60:.1f} minutes)")
    print(f"{'=' * 60}")
    for v, status in results.items():
        print(f"  Variant {v} ({VARIANTS[v]}): {status}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    subprocess.run([sys.executable, 'evaluate.py',
                    '--data_dir', args.data_dir,
                    '--output_dir', args.output_dir])


if __name__ == '__main__':
    main()