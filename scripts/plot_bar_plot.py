import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Set the figure size and style
plt.figure(figsize=(16, 5))
plt.style.use('seaborn-v0_8-whitegrid')

# Data setup
benchmarks = ['Spec-Bench', 'AgenticSQL', 'SWE-Bench']
baselines = ['Vanilla', 'Eagle', 'Eagle 2', 'Eagle 3', 'PLD', 'Token Recycling', "Suffix Decoding"]

# Generate some example data (replace with your actual data)
# Each row represents a benchmark, each column a baseline
data = np.array([
    [1, 1.832, 1.82, 1.943, 1.391, 2.2, 1.558],  # Spec-Bench
    [1, 1.53, 1.746, 1.314, 2.216, 2.409, 3.896],  # AgenticSQL
    [1, np.nan, np.nan,np.nan,  1.43, 1.328, 1.928],  # SWE-Bench
    #0.781, 1.149, 0.553,
])

# Example: set some values to np.nan to demonstrate
# (You can adjust which values are nan as needed)
# data[0, 2] = np.nan  # AgenticSQL, Baseline 3
# data[1, 4] = np.nan  # SWE-Bench, Baseline 5

# Set width of bars
bar_width = 0.12
positions = np.arange(len(benchmarks))

# Colors for different baselines - scientific publication color palette
# One color per bar (7 total)
colors = ['#0072B2', '#E69F00', '#009E73', '#56B4E9', '#CC79A7', '#D55E00', '#F0E442']

y_max = 1.0

# Plot bars or red x for nan
for i, baseline in enumerate(baselines):
    offset = (i - len(baselines)/2 + 0.5) * bar_width
    for j, value in enumerate(data[:, i]):
        x_pos = positions[j] + offset
        if np.isnan(value):
            plt.plot(x_pos, 0.2, 'x', color='red', markersize=16, markeredgewidth=3, label=None)
        else:
            plt.bar(x_pos, value, bar_width, label=baseline if j == 0 else None, color=colors[i], edgecolor='black', linewidth=1)

# Customize plot
plt.xlabel('Benchmarks', fontsize=14, fontweight='bold')
plt.ylabel('Speedup', fontsize=14, fontweight='bold')
plt.title('Speculative Speedups over Vanilla Decoding', fontsize=16, fontweight='bold')
plt.xticks(positions, benchmarks, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)
plt.legend(title='Baselines', fontsize=12, title_fontsize=13)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a subtle box around the plot
plt.box(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('benchmark_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot (optional)
plt.show()
