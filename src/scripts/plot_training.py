"""
Plot training metrics from checkpoint files.

Usage:
    python src/scripts/plot_training.py --checkpoint_dir checkpoints/frozen_transfer
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def extract_metrics_from_checkpoints(checkpoint_dir):
    """Extract training metrics from checkpoint files"""
    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')))
    
    epochs = []
    train_losses = []
    train_mious = []
    val_mious = []
    
    for ckpt_path in checkpoint_files:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'epoch' in ckpt:
                epochs.append(ckpt['epoch'])
                train_losses.append(ckpt.get('train_loss', None))
                train_mious.append(ckpt.get('train_mIoU', None))
                val_mious.append(ckpt.get('val_mIoU', None))
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
    
    return epochs, train_losses, train_mious, val_mious


def plot_metrics(checkpoint_dir, output_path=None):
    """Plot training metrics"""
    epochs, train_losses, train_mious, val_mious = extract_metrics_from_checkpoints(checkpoint_dir)
    
    if not epochs:
        print(f"No checkpoint data found in {checkpoint_dir}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if any(train_losses):
        axes[0].plot(epochs, train_losses, 'b-', marker='o', label='Train Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot mIoU
    if any(train_mious) or any(val_mious):
        if any(train_mious):
            axes[1].plot(epochs, train_mious, 'g-', marker='o', label='Train mIoU')
        if any(val_mious):
            axes[1].plot(epochs, val_mious, 'r-', marker='s', label='Val mIoU')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('mIoU', fontsize=12)
        axes[1].set_title('Mean Intersection over Union', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")
    else:
        plt.savefig(os.path.join(checkpoint_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {os.path.join(checkpoint_dir, 'training_metrics.png')}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total epochs: {len(epochs)}")
    if train_losses and train_losses[-1] is not None:
        print(f"Final train loss: {train_losses[-1]:.4f}")
    if train_mious and train_mious[-1] is not None:
        print(f"Final train mIoU: {train_mious[-1]:.4f}")
    if val_mious and val_mious[-1] is not None:
        print(f"Final val mIoU: {val_mious[-1]:.4f}")
        best_val = max([m for m in val_mious if m is not None])
        best_epoch = epochs[val_mious.index(best_val)]
        print(f"Best val mIoU: {best_val:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: checkpoint_dir/training_metrics.png)')
    
    args = parser.parse_args()
    plot_metrics(args.checkpoint_dir, args.output)

