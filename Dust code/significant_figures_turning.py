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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_value = 2025
set_random_seed(seed_value)

REAL_DATA = True
realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/root/Virtual_Data_Generation'  # project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'timestamp','operation']
new_columns = selected_columns[:6] + [selected_columns[-1]]

print('Randomly Split the real dataset into train, validation and test sets: %s' % str([0.7, 0.1, 0.2]))
print('Select acceleration data of both wrists: %s' % selected_columns)
print('Data for train, validation, and test: %s' % new_columns)

train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109',
               'U0111', 'U0201', 'U0202', 'U0203', 'U0204',
               'U0206', 'U0207', 'U0208', 'U0210']
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
    
train_data_dict = {}
for u in train_users:
    train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

val_data_dict = {}
for u in val_users:
    val_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

test_data_dict = {}
for u in test_users:
    test_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

"""
Define jitter augmentation (sigma is the noise scale)
"""
def jitter(sample, sigma=0.6):
    noise = np.random.normal(loc=0.0, scale=sigma, size=sample.shape)
    return sample + noise

"""
Assume switch_axis is defined elsewhere.
If it isn’t, you may define a dummy function:
"""
def switch_axis(channel_data, index):
    # Dummy implementation: for example, simply return the data
    return channel_data

"""
Update the custom virtual data generation to accept sigma and round_digits.
Apply jitter to the raw data during augmentation and round the acceleration data.
Also count CSV files for evaluation.
"""
def custom_virtual_data_generation(train_data_dict, i, j, sigma, round_digits):
    total_csv_count = 0

    def custom_virtual_data_generation_algorithm(data, u, labels, counter):
        # Process left channel: axis switching
        left_aug = switch_axis(data[:, :3].transpose(), i)
        # Process right channel: axis switching
        right_aug = switch_axis(data[:, 3:].transpose(), j)
        combined = np.concatenate([left_aug.transpose(), right_aug.transpose()], axis=1)
        # Before saving, round the acceleration entries to the specified number of decimals.
        # Only round the first 6 columns (acceleration data)
        rounded_acc = np.round(combined, round_digits)
        virtual_data = np.concatenate([rounded_acc, labels], axis=1)
        df = pd.DataFrame(virtual_data, columns=new_columns)
        filename = f"{u}_aug_{counter}"
        save_virtual_data(df, filename)
        return combined

    def save_virtual_data(data_df, filename):
        nonlocal total_csv_count
        total_csv_count += 1
        data_df.to_csv(os.path.join(virt_directory, filename + '.csv'), index=False)
    number = 0
    for u, df in train_data_dict.items():
        print('Generating virtual data from user %s with sigma = %s and rounding at %d decimals.' % (u, sigma, round_digits))
        raw_data = df[selected_columns[:6]].values
        # Apply jitter augmentation using the specified sigma.
        raw_data = jitter(raw_data, sigma)
        labels = df[selected_columns[-1]].values.reshape(-1, 1)
        volume_checker = True
        aug_count = 0
        number += 1
        while volume_checker:
            custom_virtual_data_generation_algorithm(raw_data, u, labels, aug_count)
            aug_count += 1
            volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500*number/len(train_data_dict))
        
    print('Virtual data generation is done for sigma = %s with rounding at %d decimals.' % (sigma, round_digits))
    return total_csv_count

if __name__ == '__main__':
    fixed_sigma = 0.4561
    for round_digits in range(16, 3, -1):
        # Clear any previously generated CSV files.
        delete_csv_files(virt_directory)
        
        # Generate virtual data with the fixed sigma and current rounding precision.
        csv_count = custom_virtual_data_generation(train_data_dict, 0, 0, fixed_sigma, round_digits)
        
        # Include the CSV file count in the identifier passed to the evaluation.
        eval_id = f"sigma_{fixed_sigma}_i_{0}_j_{0}_round_{round_digits}_csv_{csv_count}"
        f1 = HAR_evaluation(eval_id)
        print(f"sigma: {fixed_sigma}, round_digits: {round_digits}, csv_count: {csv_count} -> f1 score: {f1}")
