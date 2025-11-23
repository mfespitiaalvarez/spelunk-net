"""
Inference script for Fast-SCNN 4-channel segmentation model.

Usage (run from project root):
    python src/scripts/inference_4ch.py \
        --checkpoint checkpoints/frozen_transfer/best_model.pth \
        --data_dir src/perception_sim/training_images \
        --num_samples 10 \
        --output_dir inference_results

This script will:
1. Load your trained model
2. Run inference on test images
3. Visualize RGB, Depth, Ground Truth, and Predictions
4. Calculate and display metrics
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Add Fast-SCNN utils to path
sys.path.insert(0, os.path.join(project_root, 'external/Fast-SCNN-pytorch'))

from src.models.fast_scnn_4ch import FastSCNN4Ch
from src.datasets.cones_balls_dataset import ConesBallsDataset
from torch.utils.data import random_split
from utils.metric import SegmentationMetric

# Color map for visualization (same as in training data generation)
COLORS = np.array([
    [0, 0, 0],       # 0: Background (black)
    [255, 0, 0],     # 1: Red cone
    [255, 127, 0],   # 2: Orange cone
    [127, 0, 127],   # 3: Purple cone
    [0, 255, 0],     # 4: Green cone
    [255, 255, 0],   # 5: Yellow cone
    [0, 0, 255],     # 6: Blue cone
    [200, 0, 0],     # 7: Red ball
    [200, 200, 0],   # 8: Yellow ball
    [0, 0, 200],     # 9: Blue ball
    [0, 200, 0],     # 10: Green ball
], dtype=np.uint8)

LABEL_NAMES = [
    'Background',
    'Red Cone', 'Orange Cone', 'Purple Cone', 'Green Cone', 'Yellow Cone', 'Blue Cone',
    'Red Ball', 'Yellow Ball', 'Blue Ball', 'Green Ball'
]


def colorize_mask(mask):
    """Convert label mask to RGB visualization"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(len(COLORS)):
        color_mask[mask == label] = COLORS[label]
    return color_mask


def visualize_prediction(rgb, depth, gt_mask, pred_mask, sample_id, output_dir, pixAcc, mIoU):
    """Create a visualization comparing ground truth and prediction"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Input and Ground Truth
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'RGB Input (Sample {sample_id})', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(depth, cmap='viridis')
    axes[0, 1].set_title(f'Depth Input (max: {depth.max():.2f}m)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    gt_colored = colorize_mask(gt_mask)
    axes[0, 2].imshow(gt_colored)
    axes[0, 2].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Prediction and Comparison
    pred_colored = colorize_mask(pred_mask)
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay prediction on RGB
    overlay = rgb.copy()
    alpha = 0.5
    for label in range(1, len(COLORS)):  # Skip background
        mask_area = pred_mask == label
        if mask_area.any():
            overlay[mask_area] = (overlay[mask_area] * (1 - alpha) + COLORS[label] * alpha).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Error visualization (wrong pixels in red)
    error_vis = rgb.copy()
    errors = gt_mask != pred_mask
    error_vis[errors] = [255, 0, 0]  # Red for errors
    axes[1, 2].imshow(error_vis)
    axes[1, 2].set_title(f'Errors in Red\npixAcc: {pixAcc:.3f}, mIoU: {mIoU:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend for labels present in this image
    unique_labels = np.unique(np.concatenate([gt_mask.flatten(), pred_mask.flatten()]))
    legend_text = "Labels:\n"
    for label in sorted(unique_labels):
        if label < len(LABEL_NAMES):
            legend_text += f"{label}: {LABEL_NAMES[label]}\n"
    
    fig.text(0.02, 0.02, legend_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_{sample_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(args):
    """Run inference on the dataset"""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load label mapping to get number of classes
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_info = json.load(f)
        num_classes = 1 + len(label_info['cones']) + len(label_info['balls'])
        print(f"Number of classes: {num_classes}")
    else:
        num_classes = 11
        print(f"Warning: Label mapping not found, using default {num_classes} classes")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    full_dataset = ConesBallsDataset(
        args.data_dir,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )
    print(f"Total samples: {len(full_dataset)}")
    
    # Apply train/val split if requested
    if args.split != 'all':
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Same seed as training!
        )
        
        if args.split == 'train':
            dataset = train_dataset
            print(f"Using TRAINING split: {len(dataset)} samples")
        else:  # val
            dataset = val_dataset
            print(f"Using VALIDATION split: {len(dataset)} samples")
    else:
        dataset = full_dataset
        print(f"Using ALL samples: {len(dataset)} samples")
    
    # Helper function to get sample_id (handles both Dataset and Subset)
    def get_sample_id(dataset, idx):
        if hasattr(dataset, 'dataset'):  # It's a Subset
            real_idx = dataset.indices[idx]
            return dataset.dataset.samples[real_idx]
        else:  # It's the original dataset
            return dataset.samples[idx]
    
    # Create model (with aux=True to match training configuration)
    print(f"\nCreating 4-channel Fast-SCNN model...")
    model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=True)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics
    metric = SegmentationMetric(num_classes)
    
    # Additional tracking for precision/recall
    class_tp = np.zeros(num_classes)  # True positives per class
    class_fp = np.zeros(num_classes)  # False positives per class
    class_fn = np.zeros(num_classes)  # False negatives per class
    
    # Confusion matrix: rows = ground truth, cols = predictions
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Select samples to visualize
    if args.num_samples > len(dataset):
        args.num_samples = len(dataset)
    
    if args.random_samples:
        np.random.seed(42)
        sample_indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    else:
        # Use evenly spaced samples
        sample_indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)
    
    print(f"\n{'='*60}")
    print(f"Running inference on {args.num_samples} samples...")
    print(f"{'='*60}")
    
    results = []
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Processing"):
            # Load sample
            rgbd, gt_mask = dataset[idx]
            sample_id = get_sample_id(dataset, idx)
            
            # Add batch dimension
            rgbd_batch = rgbd.unsqueeze(0).to(device)
            
            # Run inference
            outputs = model(rgbd_batch)
            pred_logits = outputs[0]  # Main output
            
            # Get predicted classes
            pred_mask = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy()
            
            # Update metrics
            metric.update(pred_mask[np.newaxis, ...], gt_mask_np[np.newaxis, ...])
            
            # Update precision/recall tracking and confusion matrix
            for class_id in range(num_classes):
                pred_class = (pred_mask == class_id)
                gt_class = (gt_mask_np == class_id)
                
                tp = np.sum(pred_class & gt_class)
                fp = np.sum(pred_class & ~gt_class)
                fn = np.sum(~pred_class & gt_class)
                
                class_tp[class_id] += tp
                class_fp[class_id] += fp
                class_fn[class_id] += fn
            
            # Update confusion matrix
            for gt_class in range(num_classes):
                for pred_class in range(num_classes):
                    count = np.sum((gt_mask_np == gt_class) & (pred_mask == pred_class))
                    confusion_matrix[gt_class, pred_class] += count
            
            # Calculate per-sample metrics
            sample_metric = SegmentationMetric(num_classes)
            sample_metric.update(pred_mask[np.newaxis, ...], gt_mask_np[np.newaxis, ...])
            sample_pixAcc, sample_mIoU = sample_metric.get()
            
            # Get RGB and depth for visualization
            rgb = (rgbd[:3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            depth = rgbd[3].cpu().numpy()
            
            # Visualize
            if args.visualize:
                visualize_prediction(rgb, depth, gt_mask_np, pred_mask, 
                                   sample_id, args.output_dir, sample_pixAcc, sample_mIoU)
            
            results.append({
                'sample_id': sample_id,
                'pixAcc': float(sample_pixAcc),
                'mIoU': float(sample_mIoU)
            })
    
    # Overall metrics
    pixAcc, mIoU = metric.get()
    
    # Calculate per-class IoU from metric's internal state
    IoU = 1.0 * metric.total_inter / (np.spacing(1) + metric.total_union)
    
    # Calculate per-class precision and recall
    precision = class_tp / (class_tp + class_fp + np.spacing(1))
    recall = class_tp / (class_tp + class_fn + np.spacing(1))
    f1_score = 2 * (precision * recall) / (precision + recall + np.spacing(1))
    
    # Calculate per-class accuracy (diagonal of confusion matrix / row sum)
    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        row_sum = confusion_matrix[i, :].sum()
        if row_sum > 0:
            class_accuracy[i] = confusion_matrix[i, i] / row_sum
        else:
            class_accuracy[i] = 0.0
    
    print(f"\n{'='*60}")
    print(f"Inference Results")
    print(f"{'='*60}")
    print(f"Overall Pixel Accuracy: {pixAcc:.4f}")
    print(f"Overall Mean IoU: {mIoU:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    print(f"{'-'*60}")
    for i in range(min(len(LABEL_NAMES), num_classes)):
        print(f"{LABEL_NAMES[i]:<20} {IoU[i]:>8.4f} {precision[i]:>10.4f} {recall[i]:>8.4f} {f1_score[i]:>10.4f}")
    
    # Save results
    results_summary = {
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'overall_pixAcc': float(pixAcc),
        'overall_mIoU': float(mIoU),
        'per_class_metrics': {
            LABEL_NAMES[i]: {
                'IoU': float(IoU[i]),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1_score[i]),
                'accuracy': float(class_accuracy[i])
            } for i in range(min(len(LABEL_NAMES), num_classes))
        },
        'confusion_matrix': confusion_matrix.tolist(),
        'per_sample_results': results
    }
    
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    if args.visualize:
        print(f"✓ Visualizations saved to {args.output_dir}")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained Fast-SCNN 4-channel model')
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default='src/perception_sim/training_images',
                        help='Path to dataset directory')
    
    # Inference settings
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val'],
                        help='Which split to evaluate: all, train, or val')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (must match training, default: 0.2)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--random_samples', action='store_true',
                        help='Use random samples instead of evenly spaced')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization images')
    
    # Data parameters
    parser.add_argument('--depth_scale', type=float, default=1.0,
                        help='Depth scaling factor')
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help='Maximum depth for normalization')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results and visualizations')
    
    args = parser.parse_args()
    
    run_inference(args)

