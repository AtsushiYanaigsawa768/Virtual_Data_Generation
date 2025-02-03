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

realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/Virtual_Data_Generation/'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath

# Create the directory
os.makedirs(real_directory, exist_ok=True)
print(f"Directory '{realpath}' created successfully.")

# Create the directory
os.makedirs(virt_directory, exist_ok=True)
print(f"Directory '{virtpath}' created successfully.")

# Construct the URL to the Zenodo API
api_url = f"https://zenodo.org/records/11059235"

# Send a request to the Zenodo API
response = requests.get(api_url)
response.raise_for_status()  # Check for HTTP errors

# # Parse the JSON response
# data = response.json()

# Extract the file information
download_url = f"https://zenodo.org/records/11059235/files/imu-with-operation-action-labels.zip?download=1"

# Download the file
file_response = requests.get(download_url)
file_response.raise_for_status()  # Check for HTTP errors

# Save the file
file_path = os.path.join(real_directory, 'imu-with-operation-action-labels.zip')
with open(file_path, 'wb') as f:
    f.write(file_response.content)

print(f"Downloaded to {file_path}")

# Iterate over all files in the directory
for filename in os.listdir(real_directory):
    # Construct full file path
    file_path = os.path.join(real_directory, filename)

    # Check if the file is a zip file
    if filename.endswith('.zip'):
        # Open the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Extract all contents of the zip file into the directory
            zip_ref.extractall(file_path[:-4])
            print(f"Extracted: {filename}")

        # Delete the zip file
        os.remove(file_path)
        print(f"Deleted: {filename}")

print("All zip files have been processed.")