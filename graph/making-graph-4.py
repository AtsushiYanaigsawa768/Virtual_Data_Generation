import matplotlib.pyplot as plt
import numpy as np

# Define the interpolation methods and their corresponding F1 scores.
# Removing RBF- and FFT- prefixes from the labels
methods = ['Linear Interp.', 
           'Inverse', 'Multiquadric', 'Gaussian', 'Linear', 'Cubic',
           'Hann', 'Welch', 'Blackman-Harris', 
           'PCHIP', 'Akima']
scores = [0.5790, 
          0.6358, 0.5853, 0.6084, 0.6062, 0.6188,
          0.5724, 0.5756, 0.5133,
          0.6202, 0.5774]

# Manually set x positions with gaps to visually group the methods.
x_positions = [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13]

plt.figure(figsize=(12, 6))
bars = plt.bar(x_positions, scores, width=0.8, color='skyblue', edgecolor='black')

# Set the x-ticks and labels.
plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=10)

# Annotate each bar with its F1 score.
for xpos, score in zip(x_positions, scores):
    plt.text(xpos, score + 0.005, f'{score:.4f}', ha='center', va='bottom', fontsize=9)

# Add group annotations for RBF and FFT.
plt.text(4, 0.74, 'RBF Interpolation', ha='center', fontsize=12, fontweight='bold')
plt.text(9, 0.74, 'FFT Interpolation', ha='center', fontsize=12, fontweight='bold')

# Optionally add vertical dashed lines to demarcate groups.
plt.axvline(x=1.5, color='gray', linestyle='--')
plt.axvline(x=7, color='gray', linestyle='--')
plt.axvline(x=11, color='gray', linestyle='--')

plt.xlabel('Interpolation Methods', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('Grouped Performance Comparison of Interpolation Methods', fontsize=14)
plt.ylim(0.5, 0.75)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
