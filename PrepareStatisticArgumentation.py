import os
import random
import zipfile
import requests
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import math
from scipy.interpolate import CubicSpline  # for warping
from einops import rearrange, repeat
from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from evaluation import HAR_evaluation
from delete import delete_csv_files
from StableDiffusion import StableDiffusionModel
from volumeChecker import check_virtual_volume
"""
Variable Part/Preparation
"""
def set_random_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA, set seed for GPU as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

# Set a fixed random seed
seed_value = 2025
set_random_seed(seed_value)
REAL_DATA = True
realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/root/Virtual_Data_Generation'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','timestamp','operation']
train_data_dict = {}
splits = [0.7, 0.1, 0.2]

new_columns = selected_columns[:6] + [selected_columns[-1]]

print('Randomly Split the real dataset into train, validation and test sets: %s'%str(splits))

print('Select acceleration data of both wrists: %s'%selected_columns)

print('Data for train, validation, and test: %s'%new_columns)

# Attention; When you try the new method, you must change the users information.
train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109', 'U0111', 'U0201', 'U0202', 'U0203', 'U0204', 'U0206', 'U0207', 'U0208', 'U0210']
val_users = ['U0101', 'U0209']
test_users = ['U0105', 'U0110', 'U0205', 'U0209']
user_paths = {}
for root, dirs, files in os.walk(real_directory):
    for file in files:
        if file.endswith('S0100.csv'):
            user_paths[file[:-10]] = os.path.join(root, file)
        else:
            os.remove(os.path.join(root, file))  # remove unused data
for u, d in user_paths.items():
    print('%s at: %s' % (u, d))
for u in train_users:
    # Load the CSV file with only the selected columns
    train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)
# Combine training CSV files into one DataFrame
df_train = pd.concat(train_data_dict.values(), ignore_index=True)

# Identify consecutive segments (blocks) where the same operation is continuous
# and assign a new time stamp (resetting for each block) starting at 1.
df_train['block_id'] = (df_train['operation'] != df_train['operation'].shift()).cumsum()
df_train['new_time'] = df_train.groupby('block_id').cumcount() + 1

# 1. Compute block statistics for the training data:
# Collapse each block to one row with its block length.
block_lengths = df_train.groupby('block_id').agg({
    'operation': 'first',
    'new_time': 'max'
}).rename(columns={'new_time': 'block_length'})

# Compute mean and variance of the block lengths for each operation and save into a single CSV file.
all_block_stats = block_lengths.groupby('operation')['block_length'].agg(['mean', 'var']).reset_index()
csv_filename = os.path.join(rootdir, 'block_length_stats_train.csv')
all_block_stats.to_csv(csv_filename, index=False)
print(f"Block length statistics saved to {csv_filename}")

# 2. Compute explanatory variable statistics per operation and new_time:
# Explanatory variables: all selected columns except 'operation'.
features = [col for col in selected_columns if col != 'operation']
time_stats = df_train.groupby(['operation', 'new_time'])[features].agg(['mean', 'var'])

# Flatten the MultiIndex columns.
time_stats.columns = ['{}_{}'.format(var, stat) for var, stat in time_stats.columns]
time_stats = time_stats.reset_index()

# For each operation, save corresponding statistics to a separate CSV file.
for op in time_stats['operation'].unique():
    op_time_stats = time_stats[time_stats['operation'] == op]
    csv_filename = os.path.join(rootdir, f'explanatory_stats_train_{op}.csv')
    op_time_stats.to_csv(csv_filename, index=False)
    print(f"Explanatory stats for operation {op} saved to {csv_filename}")