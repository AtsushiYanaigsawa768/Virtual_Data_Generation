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
    print('%s at: %s'% (u,d))
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
2.6 Example algorithm
"""
"""
This code implements a list of transforms for tri-axial raw-accelerometry
We assume that the input format is of size:
3 x (epoch_len * sampling_frequency)

Transformations included:
1. jitter
2. Channel shuffling: which axis is being switched
3. Horizontal flip: binary
4. Permutation: binary

This script is mostly based off from
https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.py
"""

def switch_axis(sample, choice):
    """
    Randomly switch the three axises for the raw files

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-6 for direction selection
    """
    x = sample[0, :]
    y = sample[1, :]
    z = sample[2, :]

    if choice == 0:
        return sample
    elif choice == 1:
        sample = np.stack([x, y, z], axis=0)
    elif choice == 2:
        sample = np.stack([x, z, y], axis=0)
    elif choice == 3:
        sample = np.stack([y, x, z], axis=0)
    elif choice == 4:
        sample = np.stack([y, z, x], axis=0)
    elif choice == 5:
        sample = np.stack([z, x, y], axis=0)
    elif choice == 6:
        sample = np.stack([z, y, x], axis=0)
    return sample


def flip(sample):
    """
    Flip over the actigram on the temporal scale

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-1 binary
    """
    return np.flip(sample, 1)


def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile is True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(
                minSegLength, X.shape[0] - minSegLength, nPerm - 1
            )
        )
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
        X_new[pp : pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new


def permute(sample, nPerm=4, minSegLength=10):
    """
    Distort an epoch by dividing up the sample into several segments and
    then permute them

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-1 binary
    """
    sample = np.swapaxes(sample, 0, 1)
    sample = DA_Permutation(sample, nPerm=nPerm, minSegLength=minSegLength)
    sample = np.swapaxes(sample, 0, 1)
    return sample


def is_scaling_factor_invalid(scaling_factor, min_scale_sigma):
    """
    Ensure each of the abs values of the scaling
    factors are greater than the min
    """
    for i in range(len(scaling_factor)):
        if abs(scaling_factor[i] - 1) < min_scale_sigma:
            return True
    return False


def DA_Scaling(X, sigma=0.3, min_scale_sigma=0.05):
    scaling_factor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1])
    )  # shape=(1,3)
    while is_scaling_factor_invalid(scaling_factor, min_scale_sigma):
        scaling_factor = np.random.normal(
            loc=1.0, scale=sigma, size=(1, X.shape[1])
        )
    my_noise = np.matmul(np.ones((X.shape[0], 1)), scaling_factor)
    X = X * my_noise
    return X


def scaling_uniform(X, scale_range=0.15, min_scale_diff=0.02):
    low = 1 - scale_range
    high = 1 + scale_range
    scaling_factor = np.random.uniform(
        low=low, high=high, size=(X.shape[1])
    )  # shape=(3)
    while is_scaling_factor_invalid(scaling_factor, min_scale_diff):
        scaling_factor = np.random.uniform(
            low=low, high=high, size=(X.shape[1])
        )

    for i in range(3):
        X[:, i] = X[:, i] * scaling_factor[i]

    return X


def scale(sample, scale_range=0.5, min_scale_diff=0.15):
    sample = np.swapaxes(sample, 0, 1)
    sample = scaling_uniform(
        sample, scale_range=scale_range, min_scale_diff=min_scale_diff
    )
    sample = np.swapaxes(sample, 0, 1)
    return sample


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(
        X, sigma
    )  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [
        (X.shape[0] - 1) / tt_cum[-1, 0],
        (X.shape[0] - 1) / tt_cum[-1, 1],
        (X.shape[0] - 1) / tt_cum[-1, 2],
    ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (
        np.ones((X.shape[1], 1))
        * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


def time_warp(sample,  sigma=0.2):
    sample = np.swapaxes(sample, 0, 1)
    sample = DA_TimeWarp(sample, sigma=sigma)
    sample = np.swapaxes(sample, 0, 1)
    return sample

def jitter(sample, sigma=0.01):
    """
    Jitter augmentation: add Gaussian noise to the input sample.

    This technique injects randomness by adding small perturbations,
    which can improve the model's robustness by simulating sensor noise.

    Args:
        sample (numpy array): Tri-axial sensor data of shape (3, FEATURE_SIZE).
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy array: The jittered sample.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=sample.shape)
    return sample + noise


def magnitude_warp(sample, sigma=0.2, knot=4):
    """
    Magnitude warping augmentation: apply time-varying scaling to the sample.

    A smooth random curve is generated using cubic spline interpolation.
    Each channel is multiplied by a corresponding smooth random factor
    at every time step. This simulates the change in sensor sensitivity
    or gain over time.

    Args:
        sample (numpy array): Tri-axial sensor data of shape (3, FEATURE_SIZE).
        sigma (float): Standard deviation for generating the random curve.
        knot (int): Number of knots used in the cubic spline interpolation.
    
    Returns:
        numpy array: The magnitude warped sample.
    """
    # Swap axes to work with shape (time_steps, channels)
    sample_swapped = np.swapaxes(sample, 0, 1)
    time_steps, channels = sample_swapped.shape
    # Generate uniform anchor points for the random curve
    anchors = np.linspace(0, time_steps - 1, num=knot + 2)
    # Generate random scaling factors for each channel at the anchor points
    random_factors = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, channels))
    
    warped = np.zeros(sample_swapped.shape)
    for c in range(channels):
        cs = CubicSpline(anchors, random_factors[:, c])
        # Interpolate the random curve for all time steps for channel c
        factors = cs(np.arange(time_steps))
        warped[:, c] = sample_swapped[:, c] * factors

    # Swap back to original shape (channels, time_steps)
    return np.swapaxes(warped, 0, 1)

# random_array_normal = np.random.randn(3, 100)
# warped_data = time_warp(random_array_normal, 1, sigma=0.2)
# x = np.linspace(0, 100, 100)

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# # First subplot
# axs[0].plot(x, random_array_normal[0,:], label='x')
# axs[0].plot(x, random_array_normal[1,:], label='y')
# axs[0].plot(x, random_array_normal[2,:], label='z')
# axs[0].set_title('Raw data')
# axs[0].set_xlabel('timesteps')
# axs[0].set_ylabel('Value')
# axs[0].legend()

# # Second subplot
# axs[1].plot(warped_data[0,:], label='x')
# axs[1].plot(warped_data[1,:], label='y')
# axs[1].plot(warped_data[2,:], label='z')
# axs[1].set_title('Generated data')
# axs[1].set_xlabel('timesteps')
# axs[1].set_ylabel('Value')

# # Adjust layout for better spacing
# plt.tight_layout()

# # Display the plots
# plt.show()


"""
2.8 Implement data augmentation with training data and save them to folder
"""
def custom_virtual_data_generation(train_data_dict,i,j):
    '''
    This function aims to generate virtual and from train_data_dict, and save the data to virt_directory.
    Participants could not change the input and output of this function.
    Participants could modify the code inside this function.
    During the code submission, participants need to submit this function and its relavant functions, such as custom_virtual_data_generation_algorithm.
    '''
    def custom_virtual_data_generation_algorithm(data, u, labels, counter):
        '''
        Generate one augmented sample and save it directly to CSV
        :param data: numpy array, shape is (data length, dim=6)
        :param u: user ID string
        :param labels: labels array
        :param counter: counter for file naming
        '''
        # Process left channel: axis switching
        left_aug = switch_axis(data[:, :3].transpose(), i)
        # Process right channel: axis switching
        right_aug = switch_axis(data[:, 3:].transpose(), j)
        # Concatenate the two channels (after transposing back)
        combined = np.concatenate([left_aug.transpose(), right_aug.transpose()], axis=1)
        
        # Combine with labels
        virtual_data = np.concatenate([combined, labels], axis=1)
        
        # Convert to dataframe
        df = pd.DataFrame(virtual_data, columns=new_columns)
        
        # Save to CSV with unique name
        filename = f"{u}_aug_{counter}"
        save_virtual_data(df, filename)
        
        return combined

    def save_virtual_data(data, filename):
        '''
        Save CSV data to /data/virtual/
        :param data: dataframe type
        :param filename: name of the file without extension
        '''
        data.to_csv(os.path.join(virt_directory, filename + '.csv'), index=False)
    number = 0
    for u, df in train_data_dict.items():
        number += 1
        print('Generating virtual data from user %s.' % u)
        # Extract sensor data and labels
        raw_data = df[selected_columns[:6]].values
        labels = df[selected_columns[-1]].values.reshape(-1,1)
        volume_checker = True
        aug_count = 0
        while volume_checker:
            custom_virtual_data_generation_algorithm(raw_data, u, labels, aug_count)
            aug_count += 1
            volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500*number/len(train_data_dict))

    print('Virtual data generation is done.')

def function1():
    # Create or load results CSV file
    results_file = 'augmentation_results.csv'
    if not os.path.exists(results_file):
        results_df = pd.DataFrame(columns=['method', 'f1_score'])
    else:
        results_df = pd.read_csv(results_file)
    
    for i in range(7):
        for j in range(7):
            delete_csv_files(virt_directory)
            custom_virtual_data_generation(train_data_dict,i,j)
            print('Virtual data generation is done.')
            f1score = HAR_evaluation(f"switch_axis_right_{i}_left_{j}")
            print(f"switch_axis_right_{i}_left_{j} f1 score: {f1score}")
            
            # Record results
            new_row = pd.DataFrame({
                'method': [f"switch_axis_right_{i}_left_{j}"],
                'f1_score': [f1score]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Save results after each iteration
            results_df.to_csv(results_file, index=False)

def trial_randam_int(train_data_dict):
    '''
    This function aims to generate virtual data from train_data_dict with random axis switching,
    and save the data to virt_directory.
    '''
    def custom_virtual_data_generation_algorithm(data, u, labels, counter):
        '''
        Generate one augmented sample and save it directly to CSV
        :param data: numpy array, shape is (data length, dim=6)
        :param u: user ID string
        :param labels: labels array
        :param counter: counter for file naming
        '''
        i = random.randint(0, 6)
        j = random.randint(0, 6)
        # Process left channel: axis switching
        left_aug = switch_axis(data[:, :3].transpose(), i)
        # Process right channel: axis switching
        right_aug = switch_axis(data[:, 3:].transpose(), j)
        # Concatenate the two channels (after transposing back)
        combined = np.concatenate([left_aug.transpose(), right_aug.transpose()], axis=1)
        
        # Combine with labels
        virtual_data = np.concatenate([combined, labels], axis=1)
        
        # Convert to dataframe
        df = pd.DataFrame(virtual_data, columns=new_columns)
        
        # Save to CSV with unique name
        filename = f"{u}_aug_{counter}"
        save_virtual_data(df, filename)
        
        return combined

    def save_virtual_data(data, filename):
        '''
        Save CSV data to /data/virtual/
        :param data: dataframe type
        :param filename: name of the file without extension
        '''
        data.to_csv(os.path.join(virt_directory, filename + '.csv'), index=False)

    number = 0
    for u, df in train_data_dict.items():
        number += 1
        print('Generating virtual data from user %s.' % u)
        # Extract sensor data and labels
        raw_data = df[selected_columns[:6]].values
        labels = df[selected_columns[-1]].values.reshape(-1,1)
        volume_checker = True
        aug_count = 0
        while volume_checker:
            custom_virtual_data_generation_algorithm(raw_data, u, labels, aug_count)
            aug_count += 1
            volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500*number/len(train_data_dict))

    print('Virtual data generation is done.')

def function2():
    # Create or load results CSV file
    results_file = 'augmentation_results.csv'
    if not os.path.exists(results_file):
        print('No results file found.')
        results_df = pd.DataFrame(columns=['method', 'f1_score'])
    else:
        print('Results file found.')
        results_df = pd.read_csv(results_file)
    
    for trail_times in range(5):
        delete_csv_files(virt_directory)
        trial_randam_int(train_data_dict)
        print('Virtual data generation is done.')
        f1score = HAR_evaluation(f"switch_axis_right_left_random_{trail_times}")
        print(f"switch_axis_right_left_random f1 score: {f1score}")
        
        # Record results
        new_row = pd.DataFrame({
            'method': [f"switch_axis_right_left_random_{trail_times}"],
            'f1_score': [f1score]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        
        # Save results after each iteration
        results_df.to_csv(results_file, index=False)

def function0():
    # Create or load results CSV file
    results_file = 'augmentation_results.csv'
    if not os.path.exists(results_file):
        results_df = pd.DataFrame(columns=['method', 'f1_score'])
    else:
        results_df = pd.read_csv(results_file)
    
    for i in [30]:
        delete_csv_files(virt_directory)
        StableDiffusionModel(i)
        print('Virtual data generation is done.')
        f1score = HAR_evaluation(f"StableDiffusionModel_{i}")
        print(f"StableDiffusionModel_{i} f1 score: {f1score}")
        new_row = pd.DataFrame({
            'method': [f"StableDiffusionModel_{i}"],
            'f1_score': [f1score]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(results_file, index=False)

function_list = [ jitter,]
function_name = [ 'jitter_test', ]

def trial_various_function(train_data_dict, right, left):
    '''
    This function aims to generate virtual data from train_data_dict using specified augmentation functions,
    and save the data to virt_directory.
    '''
    def custom_virtual_data_generation_algorithm(data, u, labels, counter):
        '''
        Generate one augmented sample and save it directly to CSV
        :param data: numpy array, shape is (data length, dim=6)
        :param u: user ID string
        :param labels: labels array
        :param counter: counter for file naming
        '''
        # Process left channel with left augmentation function
        left_aug = left(data[:, :3].transpose())
        # Process right channel with right augmentation function
        right_aug = right(data[:, 3:].transpose())
        # Concatenate the two channels (after transposing back)
        combined = np.concatenate([left_aug.transpose(), right_aug.transpose()], axis=1)
        
        # Combine with labels
        virtual_data = np.concatenate([combined, labels], axis=1)
        
        # Convert to dataframe
        df = pd.DataFrame(virtual_data, columns=new_columns)
        
        # Save to CSV with unique name
        filename = f"{u}_aug_{counter}"
        save_virtual_data(df, filename)
        
        return combined

    def save_virtual_data(data, filename):
        '''
        Save CSV data to /data/virtual/
        :param data: dataframe type
        :param filename: name of the file without extension
        '''
        data.to_csv(os.path.join(virt_directory, filename + '.csv'), index=False)

    number = 0
    for u, df in train_data_dict.items():
        number += 1
        print('Generating virtual data from user %s.' % u)
        # Extract sensor data and labels
        raw_data = df[selected_columns[:6]].values
        labels = df[selected_columns[-1]].values.reshape(-1,1)
        volume_checker = True
        aug_count = 0
        while volume_checker:
            custom_virtual_data_generation_algorithm(raw_data, u, labels, aug_count)
            aug_count += 1
            volume_checker = check_virtual_volume(rootdir, virtpath, limit_mb=500*number/len(train_data_dict))

    print('Virtual data generation is done.')

def function3():
    # Create or load results CSV file
    results_file = 'augmentation_results.csv'
    if not os.path.exists(results_file):
        results_df = pd.DataFrame(columns=['method', 'f1_score'])
    else:
        results_df = pd.read_csv(results_file)
    
    for i in range(1):
        delete_csv_files(virt_directory)
        trial_various_function(train_data_dict, function_list[i], function_list[i])
        print('Virtual data generation is done.')
        f1score = HAR_evaluation(f"{function_name[i]}")
        print(f"{function_name[i]} f1 score: {f1score}")
        new_row = pd.DataFrame({
            'method': [f"{function_name[i]}"],
            'f1_score': [f1score]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(results_file, index=False)

if __name__ == '__main__':
    function3()