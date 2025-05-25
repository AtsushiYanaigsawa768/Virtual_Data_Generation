import matplotlib.pyplot as plt
import numpy as np

# データ拡張手法のF1スコア (Table: Performance Comparison of Various Data Augmentation Techniques)
methods = ['Jitter', 'Flip', 'Permute', 'Scale', 'Time Warp', 'Magnitude Warp']
f1_scores = [0.6351, 0.1619, 0.5811, 0.5791, 0.5528, 0.5779]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, f1_scores)
plt.xlabel('Data Augmentation Method')
plt.ylabel('F1 Score')
plt.title('Performance Comparison of Data Augmentation Techniques')
plt.ylim(0, 0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
