"""Generate training time comparison figure including deep learning methods."""
import matplotlib.pyplot as plt
import numpy as np

# Training time data from CSV files (in seconds)
# Traditional and Riemannian methods from 20260309_153353.csv
# Deep learning methods from 20260309_142507.csv

algorithms = [
    'CSP+LDA', 'CSP+SVM', 'MDM', 
    'Riemann\nTangentSpace', 'RiemannTS\n+SVM', 'RiemannTS\n+PCA', 'RiemannTS\n+RF',
    'FBCSP', 'FilterBank\nTangentSpace+SVM',
    'EEGNet', 'EEGTCNet', 'IFNet', 'EEGITNet'
]

train_times = [
    0.68, 0.67, 0.34,  # CSP+LDA, CSP+SVM, MDM
    0.40, 0.40, 0.43, 0.65,  # Riemannian methods
    9.62, 5.59,  # FBCSP, FilterBankTangentSpace+SVM
    42.58, 63.68, 76.28, 83.37  # Deep learning methods
]

# Color coding by method category
colors = [
    '#4CAF50', '#4CAF50', '#4CAF50',  # Traditional (green)
    '#2196F3', '#2196F3', '#2196F3', '#2196F3',  # Riemannian (blue)
    '#FF9800', '#FF9800',  # Filter Bank (orange)
    '#F44336', '#F44336', '#F44336', '#F44336'  # Deep Learning (red)
]

# Edge colors - highlight FBTS
edge_colors = ['black'] * len(algorithms)
edge_colors[8] = 'red'  # FBTS+SVM index
edge_widths = [0.5] * len(algorithms)
edge_widths[8] = 3  # Thicker border for FBTS

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

x_pos = np.arange(len(algorithms))
bars = ax.bar(x_pos, train_times, color=colors, edgecolor=edge_colors, linewidth=edge_widths)

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars, train_times)):
    height = bar.get_height()
    if time < 10:
        label = f'{time:.2f}s'
    else:
        label = f'{time:.1f}s'
    # Highlight FBTS label
    if i == 8:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize axes
ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison: Traditional vs Riemannian vs Deep Learning Methods\n(BCI IV 2A Dataset, 9 Subjects)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
ax.set_yscale('log')
ax.set_ylim(0.1, 200)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', edgecolor='black', label='Traditional Methods'),
    Patch(facecolor='#2196F3', edgecolor='black', label='Riemannian Methods'),
    Patch(facecolor='#FF9800', edgecolor='black', label='Filter Bank Methods'),
    Patch(facecolor='#F44336', edgecolor='black', label='Deep Learning Methods'),
    Patch(facecolor='#FF9800', edgecolor='red', linewidth=2, label='FBTS+SVM (Proposed)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('d:/EEG/openCode/MI_realtime_TangentSpace/figures/training_time_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('d:/EEG/openCode/MI_realtime_TangentSpace/figures/training_time_comparison.pdf', 
            bbox_inches='tight', facecolor='white')
print("Training time comparison figure saved to figures/training_time_comparison.png and .pdf")
