"""
Quick inference test on real flight data.

Usage:
    python src/scripts/test_flight_inference.py \
        --checkpoint checkpoints/flight_run_1/best_model.pth \
        --num_samples 10 \
        --output_dir inference_results
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external/Fast-SCNN-pytorch'))

from src.models.fast_scnn_4ch import FastSCNN4Ch
from src.datasets.flight_dataset import FlightDataset
from src.datasets.transforms import get_validation_augmentation
from utils.metric import SegmentationMetric


def visualize_sample(rgbd, gt_mask, pred_mask, class_names, sample_idx, output_path):
    """Create visualization for a single sample"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract RGB and depth
    rgb = rgbd[:3].permute(1, 2, 0).cpu().numpy()
    depth = rgbd[3].cpu().numpy()
    
    # Row 1: Inputs and GT
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'RGB Input (Sample {sample_idx})', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(depth, cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title('Depth Input (normalized)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_mask, cmap='tab20', vmin=0, vmax=len(class_names)-1)
    axes[0, 2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Prediction and analysis
    axes[1, 0].imshow(pred_mask, cmap='tab20', vmin=0, vmax=len(class_names)-1)
    axes[1, 0].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay prediction on RGB
    overlay = (rgb * 255).astype(np.uint8).copy()
    alpha = 0.4
    # Create colored overlay for non-background predictions
    colored_pred = plt.cm.tab20(pred_mask / max(len(class_names)-1, 1))[:, :, :3]
    mask_non_bg = pred_mask > 0
    overlay[mask_non_bg] = (overlay[mask_non_bg] * (1 - alpha) + 
                             colored_pred[mask_non_bg] * 255 * alpha).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Error visualization
    errors = gt_mask != pred_mask
    error_vis = (rgb * 255).astype(np.uint8).copy()
    error_vis[errors] = [255, 0, 0]  # Red for errors
    
    # Calculate accuracy
    accuracy = (gt_mask == pred_mask).sum() / gt_mask.size
    axes[1, 2].imshow(error_vis)
    axes[1, 2].set_title(f'Errors (Red)\nAccuracy: {accuracy:.3f}', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend
    unique_labels = np.unique(np.concatenate([gt_mask.flatten(), pred_mask.flatten()]))
    legend_text = "Classes Present:\n"
    for label in sorted(unique_labels):
        if label < len(class_names):
            gt_count = (gt_mask == label).sum()
            pred_count = (pred_mask == label).sum()
            legend_text += f"{label}: {class_names[label]} (GT:{gt_count}, Pred:{pred_count})\n"
    
    fig.text(0.02, 0.02, legend_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Make sure training has completed and saved a checkpoint!")
        return
    
    # Try to load split info from checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else '.'
    split_info_path = os.path.join(checkpoint_dir, 'train_val_split.json')
    
    # Load dataset with same parameters as training
    print(f"\nLoading dataset from {args.data_root}")
    val_transform = get_validation_augmentation(crop_size=512, max_depth=20.0)
    
    # Try to load split info to get training parameters
    session_folders = None
    frame_selection = None
    val_split = 0.2
    
    if os.path.exists(split_info_path):
        import json
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        session_folders = split_info.get('session_folders')
        frame_selection = split_info.get('frame_selection')
        val_split = split_info.get('val_split', 0.2)
        print(f"✓ Loaded split info from {split_info_path}")
        print(f"  - Session folders: {session_folders}")
        print(f"  - Frame selection: {frame_selection}")
        print(f"  - Val split: {val_split}")
    else:
        print(f"Warning: Split info not found at {split_info_path}")
        print(f"  Using default parameters. Results may not match training validation set!")
    
    # Create dataset with same parameters as training
    dataset = FlightDataset(
        args.data_root,
        session_folder=session_folders,
        frame_selection=frame_selection,
        transform=val_transform,
        max_depth=20.0
    )
    
    num_classes = dataset.num_classes
    class_names = dataset.get_class_names()
    
    print(f"Dataset loaded:")
    print(f"  - {len(dataset)} samples")
    print(f"  - {num_classes} classes: {class_names}")
    
    # Use saved indices if available, otherwise recreate split
    if os.path.exists(split_info_path) and args.split != 'all':
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        
        if args.split == 'val':
            indices = split_info['val_indices']
            print(f"\nUsing VALIDATION split from training: {len(indices)} samples")
        elif args.split == 'train':
            indices = split_info['train_indices']
            print(f"\nUsing TRAINING split from training: {len(indices)} samples")
    else:
        # Fallback: recreate split with same logic (should match if dataset is identical)
        import random
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        indices = list(range(total_size))
        random.seed(42)  # Same seed as training
        random.shuffle(indices)
        
        if args.split == 'val':
            indices = indices[train_size:]
            print(f"\nUsing VALIDATION split (recreated): {len(indices)} samples")
        elif args.split == 'train':
            indices = indices[:train_size]
            print(f"\nUsing TRAINING split (recreated): {len(indices)} samples")
        else:
            indices = list(range(total_size))
            print(f"\nUsing ALL samples: {len(indices)} samples")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=True)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select samples to test
    num_samples = min(args.num_samples, len(indices))
    if args.random_samples:
        test_indices = random.sample(indices, num_samples)
    else:
        # Evenly spaced
        step = len(indices) // num_samples
        test_indices = indices[::step][:num_samples]
    
    print(f"\n{'='*60}")
    print(f"Running inference on {num_samples} samples...")
    print(f"{'='*60}\n")
    
    # Initialize metrics
    metric = SegmentationMetric(num_classes)
    
    # Additional tracking for precision/recall/confusion matrix
    class_tp = np.zeros(num_classes)
    class_fp = np.zeros(num_classes)
    class_fn = np.zeros(num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Track which classes appear in each image (for image distribution calculation)
    images_containing_class = np.zeros(num_classes, dtype=np.int64)
    per_sample_classes = []  # Store which classes appear in each sample
    
    # Run inference
    with torch.no_grad():
        for i, idx in enumerate(tqdm(test_indices, desc="Processing")):
            # Load sample
            rgbd, gt_mask = dataset[idx]
            
            # Predict
            rgbd_batch = rgbd.unsqueeze(0).to(device)
            outputs = model(rgbd_batch)
            pred_logits = outputs[0]  # Main output
            
            pred_mask = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy()
            
            # Track which classes appear in this image's ground truth
            unique_classes = np.unique(gt_mask_np)
            classes_in_image = [int(c) for c in unique_classes if c < num_classes]
            per_sample_classes.append(classes_in_image)
            
            # Update images_containing_class counter
            for class_id in unique_classes:
                if class_id < num_classes:
                    images_containing_class[int(class_id)] += 1
            
            # Update metrics
            metric.update(pred_mask[np.newaxis, ...], gt_mask_np[np.newaxis, ...])
            
            # Update precision/recall tracking
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
            
            # Visualize
            if args.visualize:
                output_path = os.path.join(args.output_dir, f'sample_{i:03d}_idx_{idx}.png')
                visualize_sample(rgbd, gt_mask_np, pred_mask, class_names, idx, output_path)
    
    # Calculate metrics
    pixAcc, mIoU = metric.get()
    IoU = 1.0 * metric.total_inter / (np.spacing(1) + metric.total_union)
    precision = class_tp / (class_tp + class_fp + np.spacing(1))
    recall = class_tp / (class_tp + class_fn + np.spacing(1))
    f1_score = 2 * (precision * recall) / (precision + recall + np.spacing(1))
    
    # Calculate per-class accuracy
    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        row_sum = confusion_matrix[i, :].sum()
        if row_sum > 0:
            class_accuracy[i] = confusion_matrix[i, i] / row_sum
        else:
            class_accuracy[i] = 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Inference Results")
    print(f"{'='*60}")
    print(f"Pixel Accuracy: {pixAcc:.4f}")
    print(f"Mean IoU: {mIoU:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Accuracy':>10}")
    print(f"{'-'*70}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {IoU[i]:>8.4f} {precision[i]:>10.4f} {recall[i]:>8.4f} {f1_score[i]:>10.4f} {class_accuracy[i]:>10.4f}")
    
    # Calculate image percentages (percentage of images containing each class)
    num_samples = len(test_indices)
    images_containing_class_percent = {}
    for i, class_name in enumerate(class_names):
        if num_samples > 0:
            images_containing_class_percent[class_name] = 100.0 * images_containing_class[i] / num_samples
        else:
            images_containing_class_percent[class_name] = 0.0
    
    # Save results
    results_summary = {
        'checkpoint': args.checkpoint,
        'num_samples': len(test_indices),
        'split': args.split,
        'overall_pixAcc': float(pixAcc),
        'overall_mIoU': float(mIoU),
        'per_class_metrics': {
            class_names[i]: {
                'IoU': float(IoU[i]),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1_score[i]),
                'accuracy': float(class_accuracy[i])
            } for i in range(num_classes)
        },
        'confusion_matrix': confusion_matrix.tolist(),
        'images_containing_class': images_containing_class_percent,
        'per_sample_classes': per_sample_classes  # Store which classes appear in each sample
    }
    
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    print(f"\n{'='*60}")
    if args.visualize:
        print(f"✓ Visualizations saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test inference on flight data')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='training_data',
                        help='Root directory of flight data')
    parser.add_argument('--split', type=str, default='val', choices=['all', 'train', 'val'],
                        help='Which split to test on')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--random_samples', action='store_true',
                        help='Use random samples instead of evenly spaced')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args)

