"""
Quick and dirty script to plot inference timing statistics
"""

import matplotlib.pyplot as plt
import numpy as np

# Hardcoded timing statistics
timing_stats = {
    'mean': 20.381,
    'min': 10.934,
    'q1': 14.825,
    'median': 19.996,
    'q3': 24.986,
    'max': 47.968,
    'stddev': 6.385,
    'count': 1000
}

# Set style
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Box plot representation of timing
ax1 = axes[0]
box_data = [
    timing_stats['min'],
    timing_stats['q1'],
    timing_stats['median'],
    timing_stats['q3'],
    timing_stats['max']
]

# Create box plot manually
positions = [1]
bp = ax1.boxplot([box_data], positions=positions, widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)

# Add mean line
ax1.axhline(timing_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {timing_stats["mean"]:.3f} ms')
ax1.axhline(timing_stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {timing_stats["median"]:.3f} ms')

# Add text annotations
ax1.text(1.3, timing_stats['min'], f'Min: {timing_stats["min"]:.3f} ms', fontsize=16, fontweight='bold')
ax1.text(1.3, timing_stats['max'], f'Max: {timing_stats["max"]:.3f} ms', fontsize=16, fontweight='bold')
ax1.text(1.3, timing_stats['q1'], f'Q1: {timing_stats["q1"]:.3f} ms', fontsize=16, fontweight='bold')
ax1.text(1.3, timing_stats['q3'], f'Q3: {timing_stats["q3"]:.3f} ms', fontsize=16, fontweight='bold')

ax1.set_ylabel('Inference Time (ms)', fontsize=18, fontweight='bold')
ax1.set_title('Inference Timing Statistics', fontsize=22, fontweight='bold', pad=15)
ax1.set_xticks([1])
ax1.set_xticklabels(['Inference Time'])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(loc='upper left', fontsize=14)
ax1.set_ylim([0, timing_stats['max'] * 1.2])

# Plot 2: Summary statistics bar chart
ax2 = axes[1]
stats_names = ['Mean', 'Median', 'Min', 'Max', 'StdDev']
stats_values = [
    timing_stats['mean'],
    timing_stats['median'],
    timing_stats['min'],
    timing_stats['max'],
    timing_stats['stddev']
]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

bars = ax2.bar(stats_names, stats_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, stats_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.3f} ms', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax2.set_ylabel('Time (ms)', fontsize=18, fontweight='bold')
ax2.set_title('Timing Statistics Summary', fontsize=22, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, max(stats_values) * 1.15])

# Add text box with additional info
info_text = f'Count: {timing_stats["count"]} samples\n'
info_text += f'StdDev: {timing_stats["stddev"]:.3f} ms'

fig.text(0.5, 0.02, info_text, ha='center', fontsize=18, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save plot
output_path = 'inference_plots/val/inference_timing.png'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"✓ Inference timing plot saved to {output_path}")

plt.close()

