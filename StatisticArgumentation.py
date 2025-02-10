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
memory_checker = True
label_number = 0
while memory_checker:
    label_number += 1

    # Read block length statistics for each operation
    block_length_stats = pd.read_csv(os.path.join(rootdir, 'Statistics/block_length_stats_train.csv'))

    generated_operations = {}

    for _, stat in block_length_stats.iterrows():
        op = int(stat['operation'])
        mu = stat['mean']
        var = stat['var']
        sigma = math.sqrt(var)
        # Sample a time from a normal distribution (ensure at least 1)
        time_length = max(1, int(np.random.normal(mu, sigma)))
        
        # Read explanatory stats for the current operation
        exp_stats_path = os.path.join(rootdir, f'Statistics/explanatory_stats_train_{op}.csv')
        exp_stats = pd.read_csv(exp_stats_path)
        
        generated_features = []
        for t in range(time_length):
            # Use modulo indexing if time_length exceeds the number of rows in exp_stats
            idx = t % len(exp_stats)
            exp_row = exp_stats.iloc[idx]
            features = []
            for col_base in ['atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z', 'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z']:
                col_mean = f"{col_base}_mean"
                col_var = f"{col_base}_var"
                mu = exp_row[col_mean]
                var = exp_row[col_var]
                sigma = math.sqrt(var)
                feat = np.random.normal(mu, sigma)
                features.append(feat)
            generated_features.append(features)
        # save as csv file
        generated_features = np.array(generated_features)
        generated_df = pd.DataFrame(generated_features, columns=['atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z', 'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z'])
        generated_df['operation'] = float(op)
        generated_df.to_csv(os.path.join(virt_directory, f'generated_{op}_{label_number}.csv'), index=False)
    memory_checker = check_virtual_volume(rootdir, virtpath)
print('Virtual data generation is completed.')


