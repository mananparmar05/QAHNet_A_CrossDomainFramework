"""
QAHNet Training Pipeline.

Supports all 6 ablation variants (A-F) with:
  - Multi-task loss (infection + severity + quality)
  - Differential learning rates (CNN vs transformer vs heads)
  - Warmup + cosine annealing scheduler
  - Early stopping on validation F1
  - Auto device detection (MPS/CUDA/CPU)
  - Checkpoint saving

Usage:
    python train.py --variant F --data_dir ./PADUFES
    python train.py --variant A --data_dir ./PADUFES  # CNN-only baseline
"""
import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import gc

from dataset import create_dataloaders
from models.qahnet import build_model


def clear_memory():
    """Free unused memory — critical for Mac."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def get_device():
    """Auto-detect best device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, device, variant, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_inf_preds, all_inf_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    num_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        inf_labels = batch['infection'].to(device)
        sev_labels = batch['severity'].to(device)
        quality = batch['quality_score'].to(device)

        # Move metadata to device
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}

        # Forward
        outputs = model(images, quality_score=quality, metadata=metadata)

        # ── Multi-task loss ──
        # Infection loss (binary)
        loss_inf = F.binary_cross_entropy_with_logits(
            outputs['infection'].squeeze(-1), inf_labels
        )

        # Severity loss (3-class)
        loss_sev = F.cross_entropy(outputs['severity'], sev_labels)

        # Quality loss (regression, if applicable)
        loss_qual = torch.tensor(0.0, device=device)
        if 'quality' in outputs:
            loss_qual = F.mse_loss(outputs['quality'].squeeze(-1), quality)

        # Combined loss
        loss = 1.0 * loss_inf + 0.5 * loss_sev + 0.3 * loss_qual

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions
        with torch.no_grad():
            inf_pred = (torch.sigmoid(outputs['infection'].squeeze(-1)) > 0.5).cpu()
            all_inf_preds.extend(inf_pred.numpy())
            all_inf_labels.extend(inf_labels.cpu().numpy())

            sev_pred = outputs['severity'].argmax(dim=1).cpu()
            all_sev_preds.extend(sev_pred.numpy())
            all_sev_labels.extend(sev_labels.cpu().numpy())

    # Metrics
    inf_f1 = f1_score(all_inf_labels, all_inf_preds, average='binary', zero_division=0)
    sev_f1 = f1_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, inf_f1, sev_f1


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_inf_preds, all_inf_labels, all_inf_probs = [], [], []
    all_sev_preds, all_sev_labels = [], []
    num_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        inf_labels = batch['infection'].to(device)
        sev_labels = batch['severity'].to(device)
        quality = batch['quality_score'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}

        outputs = model(images, quality_score=quality, metadata=metadata)

        loss_inf = F.binary_cross_entropy_with_logits(
            outputs['infection'].squeeze(-1), inf_labels
        )
        loss_sev = F.cross_entropy(outputs['severity'], sev_labels)
        loss = loss_inf + 0.5 * loss_sev
        total_loss += loss.item()
        num_batches += 1

        inf_prob = torch.sigmoid(outputs['infection'].squeeze(-1)).cpu()
        inf_pred = (inf_prob > 0.5).numpy()
        all_inf_preds.extend(inf_pred)
        all_inf_probs.extend(inf_prob.numpy())
        all_inf_labels.extend(inf_labels.cpu().numpy())

        sev_pred = outputs['severity'].argmax(dim=1).cpu()
        all_sev_preds.extend(sev_pred.numpy())
        all_sev_labels.extend(sev_labels.cpu().numpy())

    inf_f1 = f1_score(all_inf_labels, all_inf_preds, average='binary', zero_division=0)
    inf_acc = accuracy_score(all_inf_labels, all_inf_preds)
    sev_f1 = f1_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds)

    try:
        inf_auc = roc_auc_score(all_inf_labels, all_inf_probs)
    except ValueError:
        inf_auc = 0.0

    avg_loss = total_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'inf_f1': inf_f1,
        'inf_acc': inf_acc,
        'inf_auc': inf_auc,
        'sev_f1': sev_f1,
        'sev_acc': sev_acc,
    }


def train(args):
    """Main training function."""
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  QAHNet Training - Variant {args.variant}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ──
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Model ──
    model = build_model(variant=args.variant, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")

    # Freeze CNN initially
    model.freeze_cnn()

    # ── Optimizer ──
    param_groups = model.get_param_groups(
        lr_cnn=args.lr_cnn,
        lr_transformer=args.lr_transformer,
        lr_heads=args.lr_heads,
    )
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Scheduler: warmup + cosine ──
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # ── Checkpoint dir ──
    ckpt_dir = os.path.join(args.output_dir, f'variant_{args.variant}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── CSV history file ──
    csv_path = os.path.join(ckpt_dir, 'training_history.csv')
    csv_fields = [
        'epoch', 'train_loss', 'train_inf_f1', 'train_sev_f1',
        'val_loss', 'val_inf_f1', 'val_inf_acc', 'val_inf_auc',
        'val_sev_f1', 'val_sev_acc', 'lr', 'time_sec', 'best'
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ── Training loop ──
    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze CNN after warmup
        if epoch == args.freeze_epochs + 1:
            print(f"\n>> Unfreezing CNN backbone at epoch {epoch}\n")
            model.unfreeze_cnn()

        t0 = time.time()
        train_loss, train_inf_f1, train_sev_f1 = train_one_epoch(
            model, loaders['train'], optimizer, device, args.variant, epoch
        )
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, loaders['val'], device)
        val_f1_combined = (val_metrics['inf_f1'] + val_metrics['sev_f1']) / 2
        clear_memory()  # Free RAM after each epoch

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[1]['lr']  # transformer LR

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train InfF1: {train_inf_f1:.3f} SevF1: {train_sev_f1:.3f} | "
            f"Val InfF1: {val_metrics['inf_f1']:.3f} SevF1: {val_metrics['sev_f1']:.3f} "
            f"AUC: {val_metrics['inf_auc']:.3f} | "
            f"LR: {lr_current:.2e} | {elapsed:.1f}s"
        )

        # History entry
        is_best = val_f1_combined > best_val_f1
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_inf_f1': train_inf_f1,
            'train_sev_f1': train_sev_f1,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': lr_current,
            'time_sec': round(elapsed, 1),
            'best': is_best,
        }
        history.append(entry)

        # Write to CSV immediately (survives crashes)
        csv_writer.writerow(entry)
        csv_file.flush()

        # Save best model
        if is_best:
            best_val_f1 = val_f1_combined
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'variant': args.variant,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> Saved best model (combined F1: {val_f1_combined:.4f})")
        else:
            patience_counter += 1

        # Save last model checkpoint (always)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'variant': args.variant,
        }, os.path.join(ckpt_dir, 'last_model.pt'))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    csv_file.close()
    print(f"Training history saved to {csv_path}")

    # ── Final evaluation on test set ──
    print(f"\n{'='*60}")
    print(f"  Final Test Evaluation - Variant {args.variant}")
    print(f"{'='*60}")

    # Load best model
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = evaluate(model, loaders['test'], device)
    print(f"\n  Infection F1:  {test_metrics['inf_f1']:.4f}")
    print(f"  Infection ACC: {test_metrics['inf_acc']:.4f}")
    print(f"  Infection AUC: {test_metrics['inf_auc']:.4f}")
    print(f"  Severity F1:   {test_metrics['sev_f1']:.4f}")
    print(f"  Severity ACC:  {test_metrics['sev_acc']:.4f}")

    # Save results
    results = {
        'variant': args.variant,
        'best_epoch': ckpt['epoch'],
        'total_params': total_params,
        'test_metrics': test_metrics,
        'history': history,
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(ckpt_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Generate training plots ──
    try:
        plot_training_curves(history, ckpt_dir, args.variant)
    except Exception as e:
        print(f"Warning: Could not generate plots ({e})")

    print(f"\nAll results saved to {ckpt_dir}/")
    print(f"  - best_model.pt     (best checkpoint)")
    print(f"  - last_model.pt     (last checkpoint)")
    print(f"  - results.json      (full results + history)")
    print(f"  - training_history.csv (epoch-by-epoch log)")
    print(f"  - training_curves.png  (loss/F1/AUC plots)")
    return test_metrics


def plot_training_curves(history, save_dir, variant):
    """Generate and save training curve plots."""
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'QAHNet Variant {variant} - Training Curves', fontsize=14, fontweight='bold')

    # Plot 1: Loss
    axes[0, 0].plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Infection F1
    axes[0, 1].plot(epochs, [h['train_inf_f1'] for h in history], 'b-', label='Train')
    axes[0, 1].plot(epochs, [h['val_inf_f1'] for h in history], 'r-', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Infection F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Severity F1
    axes[1, 0].plot(epochs, [h['train_sev_f1'] for h in history], 'b-', label='Train')
    axes[1, 0].plot(epochs, [h['val_sev_f1'] for h in history], 'r-', label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Severity F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: AUC + LR
    ax4 = axes[1, 1]
    ax4.plot(epochs, [h['val_inf_auc'] for h in history], 'g-', label='Val AUC')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC')
    ax4.set_title('Infection AUC & Learning Rate')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    ax4b = ax4.twinx()
    ax4b.plot(epochs, [h['lr'] for h in history], 'k--', alpha=0.4, label='LR')
    ax4b.set_ylabel('Learning Rate')
    ax4b.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  >> Training curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='QAHNet Training')
    parser.add_argument('--variant', type=str, default='F',
                        choices=['A', 'B', 'C', 'D', 'E', 'F'],
                        help='Model variant (A=CNN-only, F=Full QAHNet)')
    parser.add_argument('--data_dir', type=str, default='./PADUFES',
                        help='Path to PADUFES dataset folder')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_cnn', type=float, default=1e-5)
    parser.add_argument('--lr_transformer', type=float, default=1e-4)
    parser.add_argument('--lr_heads', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()