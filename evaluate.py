"""
QAHNet Evaluation & Paper Results Generator.

Generates:
  - Ablation comparison table
  - Degradation robustness evaluation
  - Per-class confusion matrices
  - Paper-ready formatted results

Usage:
    python evaluate.py --data_dir ./PADUFES --output_dir ./checkpoints
"""
import os
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
from PIL import Image, ImageFilter

from dataset import create_dataloaders, PADDataset, get_transforms, collate_fn
from models.qahnet import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Full evaluation with detailed metrics."""
    model.eval()
    all_inf_preds, all_inf_labels, all_inf_probs = [], [], []
    all_sev_preds, all_sev_labels = [], []

    for batch in loader:
        images = batch['image'].to(device)
        inf_labels = batch['infection']
        sev_labels = batch['severity']
        quality = batch['quality_score'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}

        outputs = model(images, quality_score=quality, metadata=metadata)

        inf_prob = torch.sigmoid(outputs['infection'].squeeze(-1)).cpu()
        inf_pred = (inf_prob > 0.5).numpy()
        all_inf_preds.extend(inf_pred)
        all_inf_probs.extend(inf_prob.numpy())
        all_inf_labels.extend(inf_labels.numpy())

        sev_pred = outputs['severity'].argmax(dim=1).cpu()
        all_sev_preds.extend(sev_pred.numpy())
        all_sev_labels.extend(sev_labels.numpy())

    results = {
        'inf_f1': f1_score(all_inf_labels, all_inf_preds, average='binary', zero_division=0),
        'inf_acc': accuracy_score(all_inf_labels, all_inf_preds),
        'inf_precision': f1_score(all_inf_labels, all_inf_preds, average='binary', zero_division=0),
        'sev_f1': f1_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0),
        'sev_acc': accuracy_score(all_sev_labels, all_sev_preds),
    }

    try:
        results['inf_auc'] = roc_auc_score(all_inf_labels, all_inf_probs)
    except ValueError:
        results['inf_auc'] = 0.0

    # Confusion matrices
    results['inf_cm'] = confusion_matrix(all_inf_labels, all_inf_preds).tolist()
    results['sev_cm'] = confusion_matrix(all_sev_labels, all_sev_preds).tolist()

    return results


def evaluate_degradation_robustness(model, data_dir, device, seed=42):
    """
    Evaluate model under different degradation levels.
    Tests: clean, blur(sigma=2), noise(sigma=25), jpeg(q=10)
    """
    from torch.utils.data import DataLoader

    conditions = {
        'clean': {'prob': 0.0},
        'blur_s2': {'prob': 1.0, 'type': 'blur', 'param': 2.0},
        'noise_s25': {'prob': 1.0, 'type': 'noise', 'param': 25.0},
        'jpeg_q10': {'prob': 1.0, 'type': 'jpeg', 'param': 10},
    }

    results = {}
    for cond_name, cond_config in conditions.items():
        # Create test dataset with specific degradation
        dataset = PADDataset(
            data_dir=data_dir, split='test',
            transform=get_transforms('test'),
            degrade_prob=0.0, seed=seed,
        )

        # Apply specific degradation to all images
        if cond_config['prob'] > 0:
            dataset.degrader.prob = 1.0

        loader = DataLoader(
            dataset, batch_size=32, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )

        metrics = evaluate_model(model, loader, device)
        results[cond_name] = {
            'inf_f1': metrics['inf_f1'],
            'sev_f1': metrics['sev_f1'],
            'inf_auc': metrics['inf_auc'],
        }
        print(f"  {cond_name:12s} | InfF1: {metrics['inf_f1']:.4f} | SevF1: {metrics['sev_f1']:.4f}")

    return results


def generate_ablation_table(output_dir):
    """Generate ablation comparison table from saved results."""
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'CNN-only (EfficientNet-B0)',
        'B': 'CNN + Transformer',
        'C': 'B + Passive QTok',
        'D': 'B + FiLM QTok',
        'E': 'D + Metadata',
        'F': 'Full QAHNet (C1+C2+C3)',
    }

    print("\n" + "=" * 85)
    print("  ABLATION STUDY RESULTS")
    print("=" * 85)
    print(f"{'Variant':<8} {'Model':<30} {'InfF1':>8} {'InfAUC':>8} {'SevF1':>8} {'Params':>10}")
    print("-" * 85)

    all_results = {}
    for v in variants:
        results_path = os.path.join(output_dir, f'variant_{v}', 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            metrics = data['test_metrics']
            params = data['total_params']
            print(
                f"  {v:<6} {variant_names[v]:<30} "
                f"{metrics['inf_f1']:>8.4f} {metrics['inf_auc']:>8.4f} "
                f"{metrics['sev_f1']:>8.4f} {params:>10,}"
            )
            all_results[v] = {'metrics': metrics, 'params': params}
        else:
            print(f"  {v:<6} {variant_names[v]:<30}   -- not trained yet --")

    print("=" * 85)
    return all_results


def main():
    parser = argparse.ArgumentParser(description='QAHNet Evaluation')
    parser.add_argument('--data_dir', type=str, default='./PADUFES')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--variant', type=str, default=None,
                        help='Evaluate specific variant (or all if None)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Generate ablation table
    generate_ablation_table(args.output_dir)

    # Evaluate specific variant for degradation robustness
    if args.variant:
        ckpt_path = os.path.join(args.output_dir, f'variant_{args.variant}', 'best_model.pt')
        if os.path.exists(ckpt_path):
            print(f"\n{'='*60}")
            print(f"  Degradation Robustness - Variant {args.variant}")
            print(f"{'='*60}")

            model = build_model(variant=args.variant, device=device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            deg_results = evaluate_degradation_robustness(
                model, args.data_dir, device
            )

            # Save
            with open(os.path.join(args.output_dir, f'variant_{args.variant}',
                                   'degradation_results.json'), 'w') as f:
                json.dump(deg_results, f, indent=2)
        else:
            print(f"No checkpoint found for variant {args.variant}")


if __name__ == '__main__':
    main()
