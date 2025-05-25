import matplotlib.pyplot as plt
import numpy as np

# フィルタごとの補間手法の比較 (Table: Filter Comparison for Action-Based Grouping)
filters = ['Moving Average', 'RANSAC', 'Wavelet', 'Savitzky-Golay', 'None']
pchip_scores = [0.5906, 0.5515, 0.5851, 0.5706, 0.6202]
rbf_inverse_scores = [0.5599, 0.6034, 0.5762, 0.6379, 0.6358]

x = np.arange(len(filters))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, pchip_scores, width, label='PCHIP')
rects2 = ax.bar(x + width/2, rbf_inverse_scores, width, label='RBF (Inverse)')

ax.set_xlabel('Filter Algorithm')
ax.set_ylabel('F1 Score')
ax.set_title('Filter Comparison for Action-Based Grouping')
ax.set_xticks(x)
ax.set_xticklabels(filters)
ax.legend()
ax.set_ylim(0.5, 0.7)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
