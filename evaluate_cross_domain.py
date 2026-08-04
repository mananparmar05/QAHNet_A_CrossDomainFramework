"""
QAHNet Cross-Domain Evaluation on ISIC 2019.

Loads a trained QAHNet checkpoint (from PAD-UFES-20 training)
and evaluates it on ISIC 2019 with NO retraining.

Outputs:
  - Per-class metrics table
  - PAD vs ISIC comparison table (domain gap analysis)
  - Degradation robustness on ISIC
  - Saved JSON results + confusion matrix plot

Usage:
    # Evaluate best variant (E) on ISIC:
    python evaluate_cross_domain.py \
        --isic_dir ./ISIC2019 \
        --checkpoint ./checkpoints/variant_E/best_model.pt \
        --variant E

    # Compare all trained variants:
    python evaluate_cross_domain.py \
        --isic_dir ./ISIC2019 \
        --checkpoint_dir ./checkpoints \
        --compare_all
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

from models.qahnet import build_model
from cross_domain_dataset import (
    create_isic_loader, ISICDataset, get_isic_transform,
    ISIC_INFECTION_MAP, ISIC_SEVERITY_MAP
)
from dataset import PADDataset, get_transforms, collate_fn
from torch.utils.data import DataLoader


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Core Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_loader(model, loader, device, dataset_name=''):
    """
    Run full evaluation on any loader that returns PAD-format batches.
    Works for both PAD-UFES-20 and ISIC (same collate format).
    """
    model.eval()
    all_inf_preds, all_inf_labels, all_inf_probs = [], [], []
    all_sev_preds, all_sev_labels = [], []

    for batch in loader:
        images   = batch['image'].to(device)
        quality  = batch['quality_score'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}

        outputs = model(images, quality_score=quality, metadata=metadata)

        # Infection
        inf_prob = torch.sigmoid(outputs['infection'].squeeze(-1)).cpu().numpy()
        inf_pred = (inf_prob > 0.5).astype(int)
        all_inf_probs.extend(inf_prob)
        all_inf_preds.extend(inf_pred)
        all_inf_labels.extend(batch['infection'].numpy().astype(int))

        # Severity
        sev_pred = outputs['severity'].argmax(dim=1).cpu().numpy()
        all_sev_preds.extend(sev_pred)
        all_sev_labels.extend(batch['severity'].numpy().astype(int))

    # ── Metrics ──
    inf_f1  = f1_score(all_inf_labels, all_inf_preds, average='binary', zero_division=0)
    inf_acc = accuracy_score(all_inf_labels, all_inf_preds)
    sev_f1  = f1_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds)

    try:
        inf_auc = roc_auc_score(all_inf_labels, all_inf_probs)
    except ValueError:
        inf_auc = 0.0

    inf_cm = confusion_matrix(all_inf_labels, all_inf_preds).tolist()
    sev_cm = confusion_matrix(all_sev_labels, all_sev_preds).tolist()

    # Per-class report
    report = classification_report(
        all_inf_labels, all_inf_preds,
        target_names=['Benign', 'Malignant'],
        output_dict=True, zero_division=0
    )

    results = {
        'dataset':   dataset_name,
        'n_samples': len(all_inf_labels),
        'inf_f1':    round(inf_f1,  4),
        'inf_auc':   round(inf_auc, 4),
        'inf_acc':   round(inf_acc, 4),
        'sev_f1':    round(sev_f1,  4),
        'sev_acc':   round(sev_acc, 4),
        'inf_cm':    inf_cm,
        'sev_cm':    sev_cm,
        'per_class': report,
    }
    return results


# ── Degradation Robustness on ISIC ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_isic_degradation(model, isic_dir, device, img_size=160):
    """
    Test robustness of QAHNet on ISIC images with synthetic degradation.
    Mirrors the PAD robustness test so results are directly comparable.
    """
    from PIL import ImageFilter
    import random
    import io
    from cross_domain_dataset import ISICDataset, isic_collate_fn

    conditions = {
        'clean':      lambda img: img,
        'blur_s2':    lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
        'noise_s25':  lambda img: _add_noise(img, sigma=25),
        'jpeg_q10':   lambda img: _jpeg_compress(img, quality=10),
    }

    results = {}
    for cond_name, deg_fn in conditions.items():

        class DegradedISIC(ISICDataset):
            def __getitem__(self, idx):
                item = super().__getitem__(idx)
                # Undo tensor, degrade PIL, re-tensor
                return item  # degradation applied in transform below

        # Custom transform with degradation injected
        from torchvision import transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        class DegradedTransform:
            def __init__(self, size, deg_fn):
                self.resize = transforms.Resize((size, size))
                self.to_tensor = transforms.ToTensor()
                self.norm = normalize
                self.deg_fn = deg_fn

            def __call__(self, img):
                img = self.resize(img)
                img = self.deg_fn(img)
                return self.norm(self.to_tensor(img))

        dataset = ISICDataset(
            data_dir=isic_dir,
            transform=DegradedTransform(img_size, deg_fn),
        )
        loader = DataLoader(
            dataset, batch_size=16, shuffle=False,
            num_workers=0, collate_fn=isic_collate_fn,
        )
        metrics = evaluate_on_loader(model, loader, device, dataset_name=f'ISIC_{cond_name}')
        results[cond_name] = {
            'inf_f1':  metrics['inf_f1'],
            'inf_auc': metrics['inf_auc'],
            'sev_f1':  metrics['sev_f1'],
        }
        print(f"  {cond_name:12s} | InfF1: {metrics['inf_f1']:.4f} | "
              f"AUC: {metrics['inf_auc']:.4f} | SevF1: {metrics['sev_f1']:.4f}")

    return results


def _add_noise(img, sigma=25):
    import numpy as np
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def _jpeg_compress(img, quality=10):
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

try:
    from PIL import Image
except ImportError:
    pass


# ── Comparison Table ───────────────────────────────────────────────────────────

def print_comparison_table(pad_metrics, isic_metrics, variant):
    """Print side-by-side PAD vs ISIC results (domain gap analysis)."""
    print(f"\n{'='*70}")
    print(f"  DOMAIN GAP ANALYSIS — Variant {variant}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'PAD-UFES-20':>15} {'ISIC 2019':>15} {'Gap':>10}")
    print(f"{'-'*70}")

    metrics_to_show = [
        ('Infection F1',  'inf_f1'),
        ('Infection AUC', 'inf_auc'),
        ('Infection ACC', 'inf_acc'),
        ('Severity F1',   'sev_f1'),
    ]

    for label, key in metrics_to_show:
        pad_val  = pad_metrics.get(key, 0.0)
        isic_val = isic_metrics.get(key, 0.0)
        gap      = isic_val - pad_val
        gap_str  = f"{gap:+.4f}"
        print(f"  {label:<18} {pad_val:>15.4f} {isic_val:>15.4f} {gap_str:>10}")

    print(f"{'='*70}")

    # Domain gap summary
    avg_gap = np.mean([
        isic_metrics.get('inf_f1', 0) - pad_metrics.get('inf_f1', 0),
        isic_metrics.get('inf_auc', 0) - pad_metrics.get('inf_auc', 0),
    ])
    print(f"\n  Average domain gap (F1+AUC): {avg_gap:+.4f}")
    if avg_gap > -0.10:
        print("  ✓ STRONG generalization — gap < 10%")
    elif avg_gap > -0.20:
        print("  ~ MODERATE generalization — gap 10-20%")
    else:
        print("  ✗ LARGE domain gap — model may overfit PAD")


def print_degradation_table(pad_deg, isic_deg):
    """Compare degradation robustness across domains."""
    print(f"\n{'='*75}")
    print(f"  DEGRADATION ROBUSTNESS — PAD vs ISIC (Infection F1)")
    print(f"{'='*75}")
    print(f"{'Condition':<15} {'PAD F1':>10} {'ISIC F1':>10} {'PAD→ISIC Drop':>15}")
    print(f"{'-'*75}")

    for cond in ['clean', 'blur_s2', 'noise_s25', 'jpeg_q10']:
        pad_f1  = pad_deg.get(cond, {}).get('inf_f1', 0.0)
        isic_f1 = isic_deg.get(cond, {}).get('inf_f1', 0.0)
        drop    = isic_f1 - pad_f1
        print(f"  {cond:<13} {pad_f1:>10.4f} {isic_f1:>10.4f} {drop:>+15.4f}")

    print(f"{'='*75}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_cross_domain_results(pad_metrics, isic_metrics, deg_results, save_dir, variant):
    """Generate cross-domain comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'QAHNet Variant {variant} — Cross-Domain Evaluation (PAD → ISIC)',
            fontsize=13, fontweight='bold'
        )

        # ── Plot 1: Metric comparison bar chart ──
        metrics_labels = ['Inf F1', 'Inf AUC', 'Sev F1']
        pad_vals  = [pad_metrics['inf_f1'],  pad_metrics['inf_auc'],  pad_metrics['sev_f1']]
        isic_vals = [isic_metrics['inf_f1'], isic_metrics['inf_auc'], isic_metrics['sev_f1']]

        x = np.arange(len(metrics_labels))
        w = 0.35
        axes[0].bar(x - w/2, pad_vals,  w, label='PAD-UFES-20', color='steelblue',  alpha=0.85)
        axes[0].bar(x + w/2, isic_vals, w, label='ISIC 2019',   color='darkorange', alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_labels)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Domain Comparison')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        for i, (p, q) in enumerate(zip(pad_vals, isic_vals)):
            axes[0].text(i - w/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=8)
            axes[0].text(i + w/2, q + 0.01, f'{q:.3f}', ha='center', fontsize=8)

        # ── Plot 2: Confusion matrix (ISIC infection) ──
        cm = np.array(isic_metrics['inf_cm'])
        im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[1].set_title('ISIC Confusion Matrix\n(Infection)')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Benign', 'Malignant'])
        axes[1].set_yticklabels(['Benign', 'Malignant'])
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, str(cm[i, j]),
                             ha='center', va='center',
                             color='white' if cm[i, j] > cm.max()/2 else 'black',
                             fontsize=12)

        # ── Plot 3: Degradation robustness comparison ──
        conds = ['clean', 'blur_s2', 'noise_s25', 'jpeg_q10']
        cond_labels = ['Clean', 'Blur σ=2', 'Noise σ=25', 'JPEG q=10']

        pad_deg_f1  = [deg_results['pad'].get(c, {}).get('inf_f1', 0) for c in conds]
        isic_deg_f1 = [deg_results['isic'].get(c, {}).get('inf_f1', 0) for c in conds]

        axes[2].plot(cond_labels, pad_deg_f1,  'o-', color='steelblue',  label='PAD-UFES-20', linewidth=2)
        axes[2].plot(cond_labels, isic_deg_f1, 's-', color='darkorange', label='ISIC 2019',   linewidth=2)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_ylabel('Infection F1')
        axes[2].set_title('Degradation Robustness')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[2].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'cross_domain_variant_{variant}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  >> Plot saved to {plot_path}")

    except Exception as e:
        print(f"  [WARNING] Could not generate plot: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate_single_variant(variant, checkpoint_path, isic_dir, pad_dir,
                             output_dir, device):
    """Full cross-domain evaluation for one variant."""
    print(f"\n{'#'*65}")
    print(f"  Cross-Domain Evaluation — Variant {variant}")
    print(f"{'#'*65}")

    # ── Load model ──
    model = build_model(variant=variant, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── PAD-UFES-20 test set (in-domain baseline) ──
    print(f"\n[1/4] Evaluating on PAD-UFES-20 test set (in-domain)...")
    pad_dataset = PADDataset(
        data_dir=pad_dir, split='test',
        transform=get_transforms('test'),
        degrade_prob=0.0,
    )
    pad_loader = DataLoader(
        pad_dataset, batch_size=16, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    pad_metrics = evaluate_on_loader(model, pad_loader, device, 'PAD-UFES-20')
    print(f"  PAD  | InfF1: {pad_metrics['inf_f1']:.4f} | "
          f"AUC: {pad_metrics['inf_auc']:.4f} | SevF1: {pad_metrics['sev_f1']:.4f}")

    # ── ISIC 2019 (cross-domain) ──
    print(f"\n[2/4] Evaluating on ISIC 2019 (cross-domain)...")
    isic_loader = create_isic_loader(
        data_dir=isic_dir, batch_size=16, num_workers=0
    )
    isic_metrics = evaluate_on_loader(model, isic_loader, device, 'ISIC-2019')
    print(f"  ISIC | InfF1: {isic_metrics['inf_f1']:.4f} | "
          f"AUC: {isic_metrics['inf_auc']:.4f} | SevF1: {isic_metrics['sev_f1']:.4f}")

    # ── Domain gap table ──
    print_comparison_table(pad_metrics, isic_metrics, variant)

    # ── Degradation robustness on PAD ──
    print(f"\n[3/4] PAD degradation robustness...")
    from evaluate import evaluate_degradation_robustness
    pad_deg = evaluate_degradation_robustness(model, pad_dir, device)

    # ── Degradation robustness on ISIC ──
    print(f"\n[4/4] ISIC degradation robustness...")
    isic_deg = evaluate_isic_degradation(model, isic_dir, device)

    print_degradation_table(pad_deg, isic_deg)

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'variant':      variant,
        'pad_metrics':  pad_metrics,
        'isic_metrics': isic_metrics,
        'pad_degradation':  pad_deg,
        'isic_degradation': isic_deg,
    }
    out_path = os.path.join(output_dir, f'cross_domain_variant_{variant}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Plot ──
    plot_cross_domain_results(
        pad_metrics, isic_metrics,
        deg_results={'pad': pad_deg, 'isic': isic_deg},
        save_dir=output_dir,
        variant=variant
    )

    return results


def compare_all_variants(checkpoint_dir, isic_dir, pad_dir, output_dir, device):
    """Run cross-domain eval on all trained variants and print summary table."""
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    all_results = {}

    for v in variants:
        ckpt_path = os.path.join(checkpoint_dir, f'variant_{v}', 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"  Skipping variant {v} — no checkpoint found")
            continue
        results = evaluate_single_variant(
            variant=v,
            checkpoint_path=ckpt_path,
            isic_dir=isic_dir,
            pad_dir=pad_dir,
            output_dir=output_dir,
            device=device,
        )
        all_results[v] = results

    # ── Summary table ──
    print(f"\n\n{'='*85}")
    print(f"  CROSS-DOMAIN SUMMARY — All Variants")
    print(f"{'='*85}")
    print(f"{'Variant':<8} {'PAD F1':>10} {'PAD AUC':>10} {'ISIC F1':>10} "
          f"{'ISIC AUC':>10} {'Gap F1':>10}")
    print(f"{'-'*85}")

    for v, res in all_results.items():
        pad_f1   = res['pad_metrics']['inf_f1']
        pad_auc  = res['pad_metrics']['inf_auc']
        isic_f1  = res['isic_metrics']['inf_f1']
        isic_auc = res['isic_metrics']['inf_auc']
        gap      = isic_f1 - pad_f1
        print(f"  {v:<6} {pad_f1:>10.4f} {pad_auc:>10.4f} "
              f"{isic_f1:>10.4f} {isic_auc:>10.4f} {gap:>+10.4f}")

    print(f"{'='*85}")

    # Save summary
    summary_path = os.path.join(output_dir, 'cross_domain_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='QAHNet Cross-Domain Evaluation')
    parser.add_argument('--isic_dir',       type=str, required=True,
                        help='Path to ISIC 2019 dataset folder')
    parser.add_argument('--pad_dir',        type=str, default='./PADUFES',
                        help='Path to PAD-UFES-20 dataset folder')
    parser.add_argument('--checkpoint',     type=str, default=None,
                        help='Path to specific .pt checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Root checkpoint dir (used with --compare_all)')
    parser.add_argument('--variant',        type=str, default='E',
                        choices=['A','B','C','D','E','F'])
    parser.add_argument('--output_dir',     type=str, default='./cross_domain_results')
    parser.add_argument('--compare_all',    action='store_true',
                        help='Evaluate all trained variants')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_all:
        compare_all_variants(
            checkpoint_dir=args.checkpoint_dir,
            isic_dir=args.isic_dir,
            pad_dir=args.pad_dir,
            output_dir=args.output_dir,
            device=device,
        )
    else:
        ckpt_path = args.checkpoint or os.path.join(
            args.checkpoint_dir, f'variant_{args.variant}', 'best_model.pt'
        )
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}")
            return
        evaluate_single_variant(
            variant=args.variant,
            checkpoint_path=ckpt_path,
            isic_dir=args.isic_dir,
            pad_dir=args.pad_dir,
            output_dir=args.output_dir,
            device=device,
        )


if __name__ == '__main__':
    main()
