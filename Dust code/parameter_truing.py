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
Update the custom virtual data generation to accept sigma.
Apply jitter to the raw data during augmentation.
"""
def custom_virtual_data_generation(train_data_dict, i, j):
    candidate_sigmas = [1e-13, 1e-14, 1e-15, 1e-16,]
    def custom_virtual_data_generation_algorithm(data, u, labels, counter, sigma):
        # Process left channel: axis switching
        left_aug = switch_axis(data[:, :3].transpose(), i)
        # Process right channel: axis switching
        right_aug = switch_axis(data[:, 3:].transpose(), j)
        combined = np.concatenate([left_aug.transpose(), right_aug.transpose()], axis=1)
        virtual_data = np.concatenate([combined, labels], axis=1)
        df = pd.DataFrame(virtual_data, columns=new_columns)
        filename = f"{u}_sigma_{sigma}_aug_{counter}"
        save_virtual_data(df, filename)
        return combined

    def save_virtual_data(data_df, filename):
        data_df.to_csv(os.path.join(virt_directory, filename + '.csv'), index=False)

    for u, df in train_data_dict.items():
        # choose sigma randomly for this csv file
        sigma_local = random.choice(candidate_sigmas)
        print('Generating virtual data from user %s with sigma = %s.' % (u, sigma_local))
        raw_data = df[selected_columns[:6]].values
        # Use the locally chosen sigma.
        raw_data = jitter(raw_data, sigma_local)
        labels = df[selected_columns[-1]].values.reshape(-1, 1)
        volume_checker = True
        aug_count = 0
        while volume_checker:
            custom_virtual_data_generation_algorithm(raw_data, u, labels, aug_count, sigma_local)
            aug_count += 1
            volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500/len(train_data_dict))
    print('Virtual data generation is done using random sigma per CSV file.')

def tune_sigma():
    fixed_i = 0
    fixed_j = 0
    # remove existing virtual data
    delete_csv_files(virt_directory)
    # Generate virtual data with a random sigma for each CSV file
    custom_virtual_data_generation(train_data_dict, fixed_i, fixed_j)
    # Evaluation identifier indicates "random"
    f1 = HAR_evaluation(f"random_sigma_i_{fixed_i}_j_{fixed_j}_mixed")
    print(f"Evaluation result with random sigma per CSV file: f1 score = {f1}")
    return f1

if __name__ == '__main__':
    tune_sigma()
