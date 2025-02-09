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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
realpath = r'/data/real'
virtpath = r'/data/virtual'
rootdir = r'/root/Virtual_Data_Generation'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','timestamp','operation']
train_data_dict = {}
def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it's a broken symlink
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

total_virtual_size = get_folder_size(virt_directory)
limit = 500 * 1024 * 1024  # 500 MB in bytes

if total_virtual_size <= limit:
    print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、500MB以下です。")
else:
    print(f"合計サイズは {total_virtual_size/(1024**2):.2f} MB で、500MBを超えています。")
