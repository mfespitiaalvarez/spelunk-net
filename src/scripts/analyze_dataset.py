"""
Analyze class distribution in the cone/ball segmentation dataset.

Usage:
    python src/scripts/analyze_dataset.py \
        --data_dir src/perception_sim/training_images \
        --output_dir dataset_analysis
        
This script will:
1. Count pixels per class across the entire dataset
2. Count how many images contain each class
3. Visualize class distributions with bar plots
4. Generate statistics about class balance
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.datasets.cones_balls_dataset import ConesBallsDataset

# Color map and label names (same as inference)
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


def analyze_dataset(args):
    """Analyze class distribution in the dataset"""
    
    print(f"\n{'='*60}")
    print("Dataset Class Distribution Analysis")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = ConesBallsDataset(
        args.data_dir,
        depth_scale=1.0,
        max_depth=5.0
    )
    print(f"Total samples: {len(dataset)}")
    
    # Load label mapping to get number of classes
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_info = json.load(f)
        num_classes = 1 + len(label_info['cones']) + len(label_info['balls'])
    else:
        num_classes = 11
    
    # Initialize tracking
    total_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    images_containing_class = np.zeros(num_classes, dtype=np.int64)
    class_co_occurrence = defaultdict(lambda: defaultdict(int))  # Which classes appear together
    
    print(f"\nAnalyzing {len(dataset)} images...")
    
    # Analyze each image
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        rgbd, mask = dataset[idx]
        mask_np = mask.numpy()
        
        # Count pixels per class in this image
        unique_classes, counts = np.unique(mask_np, return_counts=True)
        
        for class_id, count in zip(unique_classes, counts):
            if class_id < num_classes:
                total_pixels_per_class[class_id] += count
                images_containing_class[class_id] += 1
        
        # Track co-occurrence (which classes appear in the same image)
        unique_classes = unique_classes[unique_classes < num_classes]
        for i, class1 in enumerate(unique_classes):
            for class2 in unique_classes[i+1:]:
                class_co_occurrence[int(class1)][int(class2)] += 1
                class_co_occurrence[int(class2)][int(class1)] += 1
    
    # Calculate statistics
    total_pixels = total_pixels_per_class.sum()
    pixel_percentages = 100.0 * total_pixels_per_class / total_pixels
    image_percentages = 100.0 * images_containing_class / len(dataset)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Class Distribution Statistics")
    print(f"{'='*60}")
    print(f"{'Class':<20} {'Images':>10} {'% Images':>10} {'Pixels':>12} {'% Pixels':>10}")
    print(f"{'-'*70}")
    
    for i in range(min(num_classes, len(LABEL_NAMES))):
        print(f"{LABEL_NAMES[i]:<20} {images_containing_class[i]:>10d} "
              f"{image_percentages[i]:>9.1f}% {total_pixels_per_class[i]:>12d} "
              f"{pixel_percentages[i]:>9.2f}%")
    
    # Create 2 plots only
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Images containing class (excluding background)
    non_bg_classes = range(1, min(num_classes, len(LABEL_NAMES)))
    
    # Plot 1: Percentage of images containing each class
    ax = axes[0]
    bars = ax.bar([LABEL_NAMES[i] for i in non_bg_classes],
                   [image_percentages[i] for i in non_bg_classes],
                   color=[COLORS[i]/255.0 for i in non_bg_classes])
    ax.set_ylabel('Percentage of Images (%)', fontsize=12)
    ax.set_title('Images Containing Each Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pixel percentage per class (excluding background)
    ax = axes[1]
    bars = ax.bar([LABEL_NAMES[i] for i in non_bg_classes],
                   [pixel_percentages[i] for i in non_bg_classes],
                   color=[COLORS[i]/255.0 for i in non_bg_classes])
    ax.set_ylabel('Percentage of Pixels (%)', fontsize=12)
    ax.set_title('Pixel Distribution Per Class (excl. Background)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'class_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved to {plot_path}")
    
    if args.show_plots:
        plt.show()
    plt.close()
    
    # Plot co-occurrence heatmap
    if args.show_cooccurrence:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Build co-occurrence matrix
        cooccur_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    cooccur_matrix[i, j] = images_containing_class[i]
                else:
                    cooccur_matrix[i, j] = class_co_occurrence[i][j]
        
        # Plot heatmap
        im = ax.imshow(cooccur_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(min(num_classes, len(LABEL_NAMES))))
        ax.set_yticks(range(min(num_classes, len(LABEL_NAMES))))
        ax.set_xticklabels([LABEL_NAMES[i] for i in range(min(num_classes, len(LABEL_NAMES)))],
                          rotation=45, ha='right')
        ax.set_yticklabels([LABEL_NAMES[i] for i in range(min(num_classes, len(LABEL_NAMES)))])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Images', rotation=270, labelpad=20)
        
        # Add text annotations (commented out to hide exact counts)
        # for i in range(min(num_classes, len(LABEL_NAMES))):
        #     for j in range(min(num_classes, len(LABEL_NAMES))):
        #         text = ax.text(j, i, int(cooccur_matrix[i, j]),
        #                      ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Class Co-occurrence Matrix\n(Number of images where classes appear together)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        cooccur_path = os.path.join(args.output_dir, 'class_cooccurrence.png')
        plt.savefig(cooccur_path, dpi=150, bbox_inches='tight')
        print(f"✓ Co-occurrence heatmap saved to {cooccur_path}")
        
        if args.show_plots:
            plt.show()
        plt.close()
    
    # Save statistics to JSON
    stats = {
        'total_images': len(dataset),
        'total_pixels': int(total_pixels),
        'num_classes': num_classes,
        'per_class_stats': {
            LABEL_NAMES[i]: {
                'class_id': i,
                'images_containing': int(images_containing_class[i]),
                'percentage_of_images': float(image_percentages[i]),
                'total_pixels': int(total_pixels_per_class[i]),
                'percentage_of_pixels': float(pixel_percentages[i])
            } for i in range(min(num_classes, len(LABEL_NAMES)))
        }
    }
    
    stats_path = os.path.join(args.output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to {stats_path}")
    
    # Calculate and print class balance metrics
    print(f"\n{'='*60}")
    print("Class Balance Analysis")
    print(f"{'='*60}")
    
    # Exclude background for balance analysis
    non_bg_pixels = total_pixels_per_class[1:]
    non_bg_images = images_containing_class[1:]
    
    print(f"Pixel distribution (excluding background):")
    print(f"  Min:  {non_bg_pixels.min():,} pixels ({100*non_bg_pixels.min()/non_bg_pixels.sum():.2f}%)")
    print(f"  Max:  {non_bg_pixels.max():,} pixels ({100*non_bg_pixels.max()/non_bg_pixels.sum():.2f}%)")
    print(f"  Mean: {non_bg_pixels.mean():,.0f} pixels")
    print(f"  Std:  {non_bg_pixels.std():,.0f} pixels")
    print(f"  Imbalance ratio (max/min): {non_bg_pixels.max() / (non_bg_pixels.min() + 1):.2f}x")
    
    print(f"\nImage distribution (excluding background):")
    print(f"  Min images:  {non_bg_images.min()}")
    print(f"  Max images:  {non_bg_images.max()}")
    print(f"  Mean images: {non_bg_images.mean():.1f}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze class distribution in segmentation dataset')
    
    parser.add_argument('--data_dir', type=str, default='src/perception_sim/training_images',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively (in addition to saving)')
    parser.add_argument('--show_cooccurrence', action='store_true', default=True,
                        help='Generate class co-occurrence heatmap')
    
    args = parser.parse_args()
    
    analyze_dataset(args)

