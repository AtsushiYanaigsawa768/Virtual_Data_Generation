import os
import pandas as pd
from pathlib import Path
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
def process_csv_files():
    # 入力と出力のディレクトリパスを設定
    input_dir = Path(real_directory)
    output_dir = Path(rootdir + '/data/converted')
    delete_csv_files(output_dir)
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力ディレクトリ内のすべてのCSVファイルを処理
    for csv_file in input_dir.glob('*.csv'):
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file)
        
        # グループを特定するための変数を初期化
        current_operation = None
        group_data = []
        group_number = 1

        # データフレームを1行ずつ処理
        for index, row in df.iterrows():
            if current_operation != row['operation']:
                # 新しいoperationが見つかった場合
                if group_data:
                    group_df = pd.DataFrame(group_data)[selected_columns]  # Only keep selected columns

                    # Adjust the timestamp: subtract the first timestamp of the group from all timestamps
                    initial_timestamp = group_df.iloc[0]['timestamp']
                    group_df['timestamp'] = group_df['timestamp'] - initial_timestamp

                    output_file = output_dir / f"{int(current_operation)}_{csv_file.stem}_group_{group_number}.csv"
                    group_df.to_csv(output_file, index=False)
                    group_number += 1
                    group_data = []
                
                current_operation = row['operation']
            
            # 現在の行をグループデータに追加
            group_data.append(row.to_dict())

        # 最後のグループを保存
        if group_data:
            group_df = pd.DataFrame(group_data)[selected_columns]  # Only keep selected columns

            # Adjust the timestamp for the last group
            initial_timestamp = group_df.iloc[0]['timestamp']
            group_df['timestamp'] = group_df['timestamp'] - initial_timestamp

            output_file = output_dir / f"{csv_file.stem}_group_{group_number}.csv"
            group_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    process_csv_files()