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
"""
Variable Part
"""
REAL_DATA = True
realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/root/Virtual_Data_Generation'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
"""
1.3 Fixed Part
"""
splits = [0.7, 0.1, 0.2]
print('Randomly Split the real dataset into train, validation and test sets: %s'%str(splits))

selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','timestamp','operation']
print('Select acceleration data of both wrists: %s'%selected_columns)

new_columns = selected_columns[:6] + [selected_columns[-1]]
print('Data for train, validation, and test: %s'%new_columns)

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

"""
2.1 Filter out un-used data
"""
user_paths = {}
for root, dirs, files in os.walk(real_directory):
    for file in files:
        if file.endswith('S0100.csv'):
            user_paths[file[:-10]] = os.path.join(root, file)
        else:
          os.remove(os.path.join(root, file))  # remove unused data
for u, d in user_paths.items():
    print('%s at: %s'% (u,d))

"""
2.3Split users to train, validation, and test sets
"""
userIDs = list(user_paths.keys())

# Shuffle the list to ensure randomness
random.shuffle(userIDs)

# Calculate the split indices
total_length = len(userIDs)
train_size = int(total_length * splits[0])  # 70% of 10
val_size = int(total_length * splits[1])  # 10% of 10
test_size = total_length - train_size - val_size  # 20% of 10

# Split the list according to the calculated sizes
train_users = np.sort(userIDs[:train_size])      # First 70%
val_users = np.sort(userIDs[train_size:train_size + val_size])  # Next 10%
test_users = np.sort(userIDs[train_size + val_size:])  # Last 20%

print('Training set: %s'%train_users)
print('Validation set: %s'%val_users)
print('Test set: %s'%test_users)


"""
2.4 Load data according to userIDs
"""
# selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','timestamp','operation']
train_data_dict = {}
for u in train_users:
    # Load the CSV file with only the selected columns
    train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

val_data_dict = {}
for u in val_users:
    # Load the CSV file with only the selected columns
    val_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

test_data_dict = {}
for u in test_users:
    # Load the CSV file with only the selected columns
    test_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

"""
2.5 Show an example of data
"""
if REAL_DATA:
    df = train_data_dict[train_users[0]]

    n = 10  # only show n timestamps on fig
    # timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
    # dates = [str(datetime.datetime.fromtimestamp(ts / 1000).replace(tzinfo=timezone_jst)) for ts in df['timestamp'].values]
    dates = df.timestamp.values
    # Select n equally spaced indices to show on the x-axis
    indices = np.linspace(0, len(dates) - 1, n, dtype=int)
    selected_dates = [dates[i] for i in indices]

    data = df[['atr01/acc_x','atr01/acc_y','atr01/acc_z']].values

    l = df['operation'].values

    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    # First subplot
    axs[0].plot(dates, data[:,0], label='x')
    axs[0].plot(dates, data[:,1], label='y')
    axs[0].plot(dates, data[:,2], label='z')
    axs[0].set_title('Raw data')
    axs[0].set_xlabel('timesteps')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    # Set x-ticks for the current subplot
    # axs[0].set_xticks(selected_dates)
    # axs[0].set_xticklabels(selected_dates, rotation=60)  # Set labels and rotate
    # axs[0].set_xlim([dates[0], dates[-1]])  # Set x-axis limits
    # axs[0].grid()

    # Second subplot
    axs[1].plot(dates, l, label='label value')
    axs[1].set_title('Operation labels')
    axs[1].set_xlabel('timesteps')
    axs[1].set_ylabel('Label ID')
    axs[1].set_xticks(selected_dates)
    axs[1].set_xticklabels(selected_dates, rotation=30)  # Set labels and rotate
    axs[1].set_xlim([dates[0], dates[-1]])  # Set x-axis limits
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    plt.show()