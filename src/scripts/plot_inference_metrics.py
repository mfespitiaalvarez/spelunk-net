"""
Visualize inference metrics from model evaluation results.

Usage:
    python src/scripts/plot_inference_metrics.py \
        --results_file inference_results_val/inference_results.json \
        --output_dir inference_plots
        
This script creates comprehensive visualizations of:
- Per-class IoU, Precision, Recall, F1-Score
- Comparison between cones and balls
- Metric correlations and trade-offs
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set style - MUCH BIGGER AND BOLDER
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 24
# Ensure colors are not overridden
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab10.colors)

# Color map matching the dataset
CLASS_COLORS = {
    'Background': [0, 0, 0],
    'Red Cone': [255, 0, 0],
    'Orange Cone': [255, 127, 0],
    'Purple Cone': [127, 0, 127],
    'Green Cone': [0, 255, 0],
    'Yellow Cone': [255, 255, 0],
    'Blue Cone': [0, 0, 255],
    'Red Ball': [200, 0, 0],
    'Yellow Ball': [200, 200, 0],
    'Blue Ball': [0, 0, 200],
    'Green Ball': [0, 200, 0],
}


def load_results(results_file):
    """Load inference results from JSON"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def plot_metrics_comparison(results, output_dir):
    """Create simplified per-class metrics visualization"""
    
    metrics = results['per_class_metrics']
    
    # Exclude background
    classes = [name for name in metrics.keys() if name != 'Background']
    
    # Extract metrics
    iou_scores = [metrics[c]['IoU'] for c in classes]
    precision_scores = [metrics[c]['precision'] for c in classes]
    recall_scores = [metrics[c]['recall'] for c in classes]
    f1_scores = [metrics[c]['f1_score'] for c in classes]
    
    # Get colors for each class
    colors = [np.array(CLASS_COLORS.get(c, [128, 128, 128])) / 255.0 for c in classes]
    
    # Create single plot - larger for bigger fonts
    fig, ax1 = plt.subplots(figsize=(20, 10))
    
    x = np.arange(len(classes))
    width = 0.2
    
    bars1 = ax1.bar(x - 1.5*width, iou_scores, width, label='IoU', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x - 0.5*width, precision_scores, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars3 = ax1.bar(x + 0.5*width, recall_scores, width, label='Recall', color='#e74c3c', alpha=0.8)
    bars4 = ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=0)
    
    ax1.set_ylabel('Score', fontsize=22, fontweight='bold')
    ax1.set_title('Per-Class Performance Metrics', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=18, prop={'weight': 'bold'})
    ax1.set_ylim([0, 1.1])  # Extra space for labels
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=16, width=2)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'metrics_visualization.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to {output_path}")
    
    plt.close()
    return fig


def plot_confusion_style_matrix(results, output_dir):
    """Create a confusion-style visualization showing metric strengths"""
    
    metrics = results['per_class_metrics']
    classes = [name for name in metrics.keys() if name != 'Background']
    
    # Create matrix: classes x metrics
    metric_names = ['IoU', 'Precision', 'Recall', 'F1-Score']
    matrix = np.zeros((len(classes), len(metric_names)))
    
    for i, class_name in enumerate(classes):
        matrix[i, 0] = metrics[class_name]['IoU']
        matrix[i, 1] = metrics[class_name]['precision']
        matrix[i, 2] = metrics[class_name]['recall']
        matrix[i, 3] = metrics[class_name]['f1_score']
    
    # Create heatmap - larger for bigger fonts
    fig, ax = plt.subplots(figsize=(12, 14))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(metric_names, fontsize=18, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=18, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                         ha="center", va="center", 
                         color="black" if matrix[i, j] > 0.5 else "white",
                         fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    ax.set_title('Per-Class Metric Heatmap', fontsize=26, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics heatmap saved to {output_path}")
    
    return fig


def plot_confusion_matrix(results, output_dir):
    """Create a confusion matrix visualization"""
    
    if 'confusion_matrix' not in results:
        print("Warning: No confusion matrix found in results. Skipping confusion matrix plot.")
        return None
    
    confusion_matrix = np.array(results['confusion_matrix'])
    metrics = results['per_class_metrics']
    
    # Get class names (include background)
    classes = list(metrics.keys())
    num_classes = len(classes)
    
    # Normalize confusion matrix to percentages (row-wise)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_matrix_norm = 100.0 * confusion_matrix / row_sums
    
    # Create figure - larger for bigger fonts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot 1: Absolute counts
    im1 = ax1.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    ax1.set_title('Confusion Matrix (Absolute Counts)', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted Class', fontsize=22, fontweight='bold')
    ax1.set_ylabel('True Class', fontsize=22, fontweight='bold')
    
    # Set ticks
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax1.set_yticklabels(classes, fontsize=16, fontweight='bold')
    
    # Add text annotations for absolute counts
    thresh = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax1.text(j, i, f'{int(confusion_matrix[i, j])}',
                         ha="center", va="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black",
                         fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Count', rotation=270, labelpad=20, fontsize=20, fontweight='bold')
    cbar1.ax.tick_params(labelsize=16)
    
    # Plot 2: Normalized percentages
    im2 = ax2.imshow(confusion_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax2.set_title('Confusion Matrix (Normalized %)', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Predicted Class', fontsize=22, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=22, fontweight='bold')
    
    # Set ticks
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax2.set_yticklabels(classes, fontsize=16, fontweight='bold')
    
    # Add text annotations for percentages
    thresh = 50.0
    for i in range(num_classes):
        for j in range(num_classes):
            if confusion_matrix[i, j] > 0:  # Only show if there are samples
                text = ax2.text(j, i, f'{confusion_matrix_norm[i, j]:.1f}%',
                             ha="center", va="center",
                             color="white" if confusion_matrix_norm[i, j] > thresh else "black",
                             fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=20, fontweight='bold')
    cbar2.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    
    plt.close()
    return fig


def plot_per_class_accuracy_recall(results, output_dir):
    """Create per-class accuracy and recall plots"""
    
    metrics = results['per_class_metrics']
    
    # Exclude background
    classes = [name for name in metrics.keys() if name != 'Background' and name != 'background']
    
    # Extract metrics
    accuracy_scores = [metrics[c].get('accuracy', 0.0) for c in classes]
    recall_scores = [metrics[c]['recall'] for c in classes]
    
    # Create figure with two subplots - larger for bigger fonts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Hardcoded colors for each class (matching actual class names)
    HARDCODED_COLORS = {
        'red_cone': '#FF0000',      # Red
        'orange_cone': '#FF7F00',   # Orange
        'purple_cone': '#7F007F',   # Purple
        'green_cone': '#00FF00',    # Green
        'yellow_cone': '#FFFF00',   # Yellow
        'blue_cone': '#0000FF',     # Blue
        'red_ball': '#C80000',      # Dark Red
        'yellow_ball': '#C8C800',   # Dark Yellow
        'blue_ball': '#0000C8',     # Dark Blue
        'green_ball': '#00C800',    # Dark Green
    }
    
    # Get colors for each class
    colors = [HARDCODED_COLORS.get(c, '#808080') for c in classes]
    
    x = np.arange(len(classes))
    width = 0.7
    
    # Plot 1: Per-class Accuracy - plot each bar individually with its color
    bars1 = []
    for i, (score, color) in enumerate(zip(accuracy_scores, colors)):
        # Explicitly set color to override any colormap
        bar = ax1.bar(i, score, width, color=color, alpha=0.95, edgecolor='black', linewidth=3, label=classes[i])
        bars1.extend(bar)
    ax1.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
    ax1.set_title('Per-Class Accuracy', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Plot 2: Per-class Recall - use same colors
    bars2 = []
    for i, (score, color) in enumerate(zip(recall_scores, colors)):
        # Explicitly set color to override any colormap
        bar = ax2.bar(i, score, width, color=color, alpha=0.95, edgecolor='black', linewidth=3, label=classes[i])
        bars2.extend(bar)
    ax2.set_ylabel('Recall', fontsize=22, fontweight='bold')
    ax2.set_title('Per-Class Recall', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_class_accuracy_recall.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Per-class accuracy/recall plot saved to {output_path}")
    
    plt.close()
    return fig


def plot_radar_charts(results, output_dir):
    """Create radar charts for cones and balls"""
    
    metrics = results['per_class_metrics']
    
    # Separate cones and balls
    cone_classes = [c for c in metrics.keys() if 'Cone' in c]
    ball_classes = [c for c in metrics.keys() if 'Ball' in c]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw=dict(projection='polar'))
    
    metric_names = ['IoU', 'Precision', 'Recall', 'F1']
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot cones
    for cone in cone_classes:
        values = [
            metrics[cone]['IoU'],
            metrics[cone]['precision'],
            metrics[cone]['recall'],
            metrics[cone]['f1_score']
        ]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=cone)
        ax1.fill(angles, values, alpha=0.15)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metric_names, fontsize=18, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.set_title('Cone Classes Performance', fontsize=26, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=16, prop={'weight': 'bold'})
    ax1.grid(True)
    ax1.tick_params(labelsize=16)
    
    # Plot balls
    for ball in ball_classes:
        values = [
            metrics[ball]['IoU'],
            metrics[ball]['precision'],
            metrics[ball]['recall'],
            metrics[ball]['f1_score']
        ]
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=3, label=ball, markersize=8)
        ax2.fill(angles, values, alpha=0.15)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names, fontsize=18, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.set_title('Ball Classes Performance', fontsize=26, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=16, prop={'weight': 'bold'})
    ax2.tick_params(labelsize=16)
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'radar_charts.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Radar charts saved to {output_path}")
    
    return fig


def plot_data_distribution(results, output_dir):
    """Create data distribution plot showing pixel percentage and image percentage per class (excluding background)"""
    
    metrics = results['per_class_metrics']
    
    # Get all classes in the order they appear in metrics (matches confusion matrix order)
    all_classes = list(metrics.keys())
    
    # Exclude background to show meaningful distribution
    classes = [name for name in all_classes if name != 'Background' and name != 'background']
    
    # Calculate pixel percentages from confusion matrix if available
    if 'confusion_matrix' in results:
        confusion_matrix = np.array(results['confusion_matrix'])
        # Sum across columns (axis=1) to get total pixels per ground truth class
        # confusion_matrix[i, j] = pixels where GT=i and Pred=j
        # Sum over j gives total pixels with GT=i
        total_pixels_per_class = confusion_matrix.sum(axis=1)
        
        # Find background index (check both cases)
        bg_idx = None
        if 'Background' in all_classes:
            bg_idx = all_classes.index('Background')
        elif 'background' in all_classes:
            bg_idx = all_classes.index('background')
        
        # Calculate percentages excluding background
        if bg_idx is not None:
            # Total pixels excluding background
            total_non_bg = total_pixels_per_class.sum() - total_pixels_per_class[bg_idx]
            # Get percentages for non-background classes
            pixel_percentages = []
            for c in classes:
                idx = all_classes.index(c)
                if total_non_bg > 0:
                    pixel_percentages.append(100.0 * total_pixels_per_class[idx] / total_non_bg)
                else:
                    pixel_percentages.append(0.0)
        else:
            # No background found, use all classes
            total_pixels = total_pixels_per_class.sum()
            pixel_percentages = [100.0 * total_pixels_per_class[all_classes.index(c)] / total_pixels 
                               if total_pixels > 0 else 0.0 for c in classes]
    else:
        # Fallback: can't calculate without confusion matrix
        print("Warning: No confusion matrix found. Cannot calculate pixel distribution.")
        return None
    
    # Calculate image percentages (percentage of images containing each class)
    num_samples = results.get('num_samples', 0)
    image_percentages = []
    
    # Use accurate data from inference script if available
    if 'images_containing_class' in results:
        # Use the accurate percentages calculated during inference
        for c in classes:
            # Handle both lowercase and title case class names
            class_key = c
            if class_key not in results['images_containing_class']:
                # Try alternative naming (e.g., 'blue_ball' vs 'Blue Ball')
                class_key_alt = c.replace('_', ' ').title()
                if class_key_alt in results['images_containing_class']:
                    class_key = class_key_alt
                else:
                    # Try lowercase
                    class_key_lower = c.lower()
                    if class_key_lower in results['images_containing_class']:
                        class_key = class_key_lower
            
            if class_key in results['images_containing_class']:
                image_percentages.append(results['images_containing_class'][class_key])
            else:
                # Fallback: class not found in results, set to 0
                image_percentages.append(0.0)
    else:
        # Fallback: estimate based on pixel counts if accurate data not available
        max_pixels = max([total_pixels_per_class[all_classes.index(c2)] for c2 in classes]) if classes else 1
        
        for c in classes:
            idx = all_classes.index(c)
            class_pixel_count = total_pixels_per_class[idx]
            if class_pixel_count > 0 and num_samples > 0:
                if max_pixels > 0:
                    # Estimate: scale between 20% and 95% of images based on pixel ratio
                    pixel_ratio = class_pixel_count / max_pixels
                    estimated_ratio = 0.2 + 0.75 * pixel_ratio
                    estimated_images = max(1, int(num_samples * estimated_ratio))
                    image_percentages.append(100.0 * estimated_images / num_samples)
                else:
                    image_percentages.append(0.0)
            else:
                image_percentages.append(0.0)
    
    # Hardcoded colors for each class (matching actual class names)
    HARDCODED_COLORS = {
        'red_cone': '#FF0000',      # Red
        'orange_cone': '#FF7F00',   # Orange
        'purple_cone': '#7F007F',   # Purple
        'green_cone': '#00FF00',    # Green
        'yellow_cone': '#FFFF00',   # Yellow
        'blue_cone': '#0000FF',     # Blue
        'red_ball': '#C80000',      # Dark Red
        'yellow_ball': '#C8C800',    # Dark Yellow
        'blue_ball': '#0000C8',     # Dark Blue
        'green_ball': '#00C800',    # Dark Green
    }
    
    # Get colors for each class
    colors = [HARDCODED_COLORS.get(c, '#808080') for c in classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    x = np.arange(len(classes))
    width = 0.7
    
    # Plot 1: Pixel percentages
    bars1 = []
    for i, (percentage, color) in enumerate(zip(pixel_percentages, colors)):
        bar = ax1.bar(i, percentage, width, color=color, alpha=0.95, edgecolor='black', linewidth=3)
        bars1.extend(bar)
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars1, pixel_percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Percentage of Pixels (%)', fontsize=22, fontweight='bold')
    ax1.set_title('Pixel Distribution Per Class', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight='bold')
    if pixel_percentages:
        ax1.set_ylim([0, max(pixel_percentages) * 1.15])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=16, width=2)
    
    # Plot 2: Image percentages
    bars2 = []
    for i, (percentage, color) in enumerate(zip(image_percentages, colors)):
        bar = ax2.bar(i, percentage, width, color=color, alpha=0.95, edgecolor='black', linewidth=3)
        bars2.extend(bar)
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars2, image_percentages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax2.set_ylabel('Percentage of Images (%)', fontsize=22, fontweight='bold')
    title_suffix = '' if 'images_containing_class' in results else ' (Estimated)'
    ax2.set_title(f'Image Distribution Per Class{title_suffix}', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight='bold')
    if image_percentages:
        ax2.set_ylim([0, max(image_percentages) * 1.15 if max(image_percentages) > 0 else 100])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=16, width=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'data_distribution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Data distribution plot saved to {output_path}")
    
    plt.close()
    return fig


def main(args):
    """Main function"""
    
    print(f"\n{'='*60}")
    print("Inference Metrics Visualization")
    print(f"{'='*60}")
    
    # Load results
    print(f"\nLoading results from {args.results_file}")
    results = load_results(args.results_file)
    
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Overall Pixel Accuracy: {results['overall_pixAcc']:.4f}")
    print(f"Overall Mean IoU: {results['overall_mIoU']:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    
    plot_metrics_comparison(results, args.output_dir)
    plot_confusion_style_matrix(results, args.output_dir)
    plot_confusion_matrix(results, args.output_dir)
    plot_per_class_accuracy_recall(results, args.output_dir)
    plot_radar_charts(results, args.output_dir)
    plot_data_distribution(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"All plots saved to {args.output_dir}")
    print(f"{'='*60}\n")
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize inference metrics')
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to inference_results.json file')
    parser.add_argument('--output_dir', type=str, default='inference_plots',
                        help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')
    
    args = parser.parse_args()
    
    main(args)

