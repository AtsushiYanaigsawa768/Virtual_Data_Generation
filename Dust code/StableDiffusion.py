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
def StableDiffusionModel(indicated_epoch=10):
    # トレーニングデータからすべてのユニークなラベルを取得
    all_labels = pd.concat(list(train_data_dict.values()))['operation'].unique()
    all_labels = np.sort(all_labels)
    all_label_list = all_labels.tolist()
    duration_label ={}
    # Load duration_label from the train data summary CSV file
    train_summary_path = os.path.join(rootdir, "Statistics", "train_data_summary_by_op.csv")
    train_summary_df = pd.read_csv(train_summary_path)
    duration_label = dict(zip(train_summary_df['operation'], train_summary_df['block_size_mean'].astype(int)))
    ########################################
    # Diffusion モデルの定義・学習・生成
    ########################################

    # 1. データセットの作成：train_data_dict のデータを用います
    def create_real_dataset():
        # 全ユーザーのデータを1つの DataFrame に連結
        all_data = pd.concat(list(train_data_dict.values()), ignore_index=True)
        # 6次元のセンサーデータ（['atr01/acc_x', ..., 'atr02/acc_z']）を抽出
        sensor_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z']
        sensor_data = all_data[sensor_columns].to_numpy().astype(np.float32)
        # normalize
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / sensor_data.std(axis=0)
        # 'operation' カラムを条件ラベルとして使用
        sensor_labels = all_data['operation'].to_numpy().astype(np.float32).reshape(-1, 1)
        return sensor_data, sensor_labels

    # ダミーデータから DataLoader を作成
    sensor_data, sensor_labels = create_real_dataset()
    diffusion_dataset = torch.utils.data.TensorDataset(torch.from_numpy(sensor_data),
                                                        torch.from_numpy(sensor_labels))
    train_loader_diff = DataLoader(diffusion_dataset, batch_size=32, shuffle=True)

    ########################################
    # 2. 時刻埋め込みレイヤ
    ########################################
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.emb_dim = emb_dim

        def forward(self, t):
            # t : (batch,) のテンソル（各要素はスカラー時刻）
            device = t.device
            half_dim = self.emb_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            return emb  # (batch, emb_dim)

    ########################################
    # 3. ノイズ除去ネットワーク（条件付き MLP）の定義
    ########################################
    class DiffusionModel(nn.Module):
        def __init__(self, data_dim, time_emb_dim=32, label_emb_dim=16):
            """
            :param data_dim: 生成対象の次元（ここでは 6 センサ値）
            """
            super().__init__()
            self.time_embed = SinusoidalPosEmb(time_emb_dim)
            self.label_embed = nn.Sequential(
                nn.Linear(1, label_emb_dim),
                nn.ReLU(),
                nn.Linear(label_emb_dim, label_emb_dim)
            )
            self.net = nn.Sequential(
                nn.Linear(data_dim + time_emb_dim + label_emb_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, data_dim)
            )

        def forward(self, x, t, label):
            # x : (batch, data_dim)
            # t : (batch,) ※ 時刻情報（拡散ステップ t）; float 型として埋め込みに使用
            # label : (batch, 1)
            t_emb = self.time_embed(t)            # (batch, time_emb_dim)
            label_emb = self.label_embed(label)     # (batch, label_emb_dim)
            h = torch.cat([x, t_emb, label_emb], dim=-1)
            return self.net(h)  # (batch, data_dim) → 予測するノイズ

    ########################################
    # 4. Diffusion プロセスの実装
    ########################################
    class Diffusion:
        def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
            """
            :param model: ノイズ除去ネットワーク（DiffusionModel）
            :param T: 拡散ステップ数
            """
            self.model = model
            self.T = T
            self.device = device
            self.beta = torch.linspace(beta_start, beta_end, T).to(device)  # (T,)
            self.alpha = 1 - self.beta  # (T,)
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # (T,)

        def q_sample(self, x0, t, noise=None):
            """
            前向き拡散過程：x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
            :param x0: 元データ (batch, data_dim)
            :param t: (batch,) の各サンプルに対する拡散ステップ（整数）
            """
            if noise is None:
                noise = torch.randn_like(x0)
            t = t.long()
            sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).unsqueeze(1)          # (batch, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).unsqueeze(1)
            return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        def p_losses(self, x0, t, label):
            """
            学習時の損失：予測したノイズと実際に加えたノイズとの MSE
            """
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)
            predicted_noise = self.model(x_t, t.float(), label)
            return nn.MSELoss()(noise, predicted_noise)

        @torch.no_grad()
        def sample(self, label, shape):
            """
            逆拡散過程により新規サンプル生成
            :param label: 条件ラベル (batch, 1)
            :param shape: 生成する x の形状、例：(batch, data_dim)
            """
            x = torch.randn(shape, device=self.device)
            for t in reversed(range(self.T)):
                t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t_batch.float(), label)
                beta_t = self.beta[t]
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) \
                    + torch.sqrt(beta_t) * noise
            return x

    ########################################
    # 5. Diffusion モデルの学習
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = 6   # センサ値は6次元
    model_diff = DiffusionModel(data_dim=data_dim).to(device)
    diffusion = Diffusion(model_diff, T=1000, device=device)
    optimizer_diff = optim.Adam(model_diff.parameters(), lr=2e-4)
    num_epochs = indicated_epoch  # 必要なエポック数に合わせて調整

    print("==== Diffusion Model Training Start ====")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_label in train_loader_diff:
            batch_x = batch_x.to(device)       # (batch, 6)
            batch_label = batch_label.to(device) # (batch, 1)
            optimizer_diff.zero_grad()
            # 各サンプルに対してランダムな拡散ステップ t を選ぶ（0 ～ T-1）
            t = torch.randint(0, diffusion.T, (batch_x.shape[0],), device=device)
            loss = diffusion.p_losses(batch_x, t, batch_label)
            loss.backward()
            optimizer_diff.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader_diff):.6f}")

    ########################################
    # 6. 各ラベル条件下でのサンプリング例：各ラベルごとにデータ生成とCSV保存（逆正則化付き）
    ########################################
    # 訓練データから各センサーチャンネルの元の平均と標準偏差を計算
    all_data = pd.concat(list(train_data_dict.values()), ignore_index=True)
    sensor_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z']
    sensor_mean = all_data[sensor_columns].mean().to_numpy().astype(np.float32)
    sensor_std = all_data[sensor_columns].std().to_numpy().astype(np.float32)
    # Tensorに変換
    sensor_mean_t = torch.tensor(sensor_mean, device=device)
    sensor_std_t = torch.tensor(sensor_std, device=device)
        
    # memory check
    memory_checker = True
    label_number = 0
    while memory_checker:
        label_number += 1
        for label_value in all_label_list:
            n_samples = duration_label[label_value]
            cond_label = torch.full((n_samples, 1), float(label_value), device=device)
            generated_x = diffusion.sample(cond_label, (n_samples, data_dim))
            # 逆正規化
            generated_x = generated_x * sensor_std_t + sensor_mean_t
            # 生成結果に条件ラベルを連結する（shape: (n_samples, 7)）
            generated_data = torch.cat([generated_x, cond_label], dim=1)
            # pandas DataFrameに変換（カラム名: 6センサ値 + operation）
            columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','operation']
            df_generated = pd.DataFrame(generated_data.cpu().numpy(), columns=columns)
            filename = f"{virt_directory}/generated_{label_value}_{label_number}.csv"
            df_generated.to_csv(filename, index=False)
        memory_checker = check_virtual_volume(rootdir, virtpath)
if __name__ == '__main__':
    StableDiffusionModel(1)