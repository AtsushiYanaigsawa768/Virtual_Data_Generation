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
"""
1.3 Fixed Part
"""
splits = [0.7, 0.1, 0.2]
print('Randomly Split the real dataset into train, validation and test sets: %s'%str(splits))

selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z','atr02/acc_x','atr02/acc_y','atr02/acc_z','timestamp','operation']
print('Select acceleration data of both wrists: %s'%selected_columns)

new_columns = selected_columns[:6] + [selected_columns[-1]]
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

def HAR_evaluation(model_name):
    # find csv files in 'data/virtual'
    virt_paths = []
    for root, dirs, files in os.walk(virt_directory):
        for file in files:
            if file.endswith('.csv'):
                virt_paths.append(os.path.join(root, file))
    print('Virtual csv file paths are as shown follows:')
    for p in virt_paths:
        print(p)
    # real and virtual training data

    ## real data
    train_data = []
    for u, data in train_data_dict.items():
        train_data.append(data[new_columns].values)
        # print(data[new_columns].values.shape)

    ## virtual data
    for p in virt_paths:
        # Load the CSV file with only the selected columns
        data = pd.read_csv(p, usecols=new_columns)
        train_data.append(data.values)

    train_data = np.concatenate(train_data, axis=0)
    print('Shape of train data is %s'%str(train_data.shape))

    # validatation and test data
    val_data = []
    for u, data in val_data_dict.items():
        val_data.append(data[new_columns].values)

    test_data = []
    for u, data in test_data_dict.items():
        test_data.append(data[new_columns].values)

    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    print('Shape of validation data is %s'%str(val_data.shape))
    print('Shape of test data is %s'%str(test_data.shape))

    # convert operation ID to labels (from 0 to n)
    labels = np.unique(train_data[:, -1])
    label_dict = dict(zip(labels, np.arange(len(labels))))
    train_data[:,-1] = np.array([label_dict[i] for i in train_data[:,-1]])
    val_data[:,-1] =  np.array([label_dict[i] for i in val_data[:,-1]])
    test_data[:,-1] =  np.array([label_dict[i] for i in test_data[:,-1]])

    class data_loader_OpenPack(Dataset):
        def __init__(self, samples, labels, device='cpu'):
            self.samples = torch.tensor(samples).to(device)  # check data type
            self.labels = torch.tensor(labels)  # check data type

        def __getitem__(self, index):
            target = self.labels[index]
            sample = self.samples[index]
            return sample, target

        def __len__(self):
            return len(self.labels)

    def sliding_window(datanp, len_sw, step):
        '''
        :param datanp: shape=(data length, dim) raw sensor data and the labels. The last column is the label column.
        :param len_sw: length of the segmented sensor data
        :param step: overlapping length of the segmented data
        :return: shape=(N, len_sw, dim) batch of sensor data segment.
        '''

        # generate batch of data by overlapping the training set
        data_batch = []
        for idx in range(0, datanp.shape[0] - len_sw - step, step):
            data_batch.append(datanp[idx: idx + len_sw, :])
        data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
        xlist = np.stack(data_batch, axis=0)  # [B, data length, dim]

        return xlist

    def generate_dataloader(data, len_sw, step, if_shuffle=True):
        tmp_b = sliding_window(data, len_sw, step)
        data_b = tmp_b[:, :, :-1]
        label_b = tmp_b[:, :, -1]
        data_set_r = data_loader_OpenPack(data_b, label_b, device=device)
        data_loader = DataLoader(data_set_r, batch_size=batch_size,
                                shuffle=if_shuffle, drop_last=False)
        return data_loader

    len_sw = 300
    step = 150
    batch_size = 512

    train_loader = generate_dataloader(train_data, len_sw, step, if_shuffle=True)
    val_loader = generate_dataloader(val_data, len_sw, step, if_shuffle=False)
    test_loader = generate_dataloader(test_data, len_sw, step, if_shuffle=False)

    class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(x, **kwargs) + x

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dropout=0.5):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5

            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x, mask=None):
            b, n, _, h = *x.shape, self.heads
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

            if mask is not None:
                mask = F.pad(mask.flatten(1), (1, 0), value=True)
                assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
                mask = mask[:, None, :] * mask[:, :, None]
                dots.masked_fill_(~mask, float('-inf'))
                del mask

            self.attn = dots.softmax(dim=-1)

            out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            return out

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(self.norm(x), **kwargs)

    class Transformer_block(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim, dropout):
            super().__init__()
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    (PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

        def forward(self, x, mask=None):
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
            return x

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            return self.dropout(x)

    class Seq_Transformer(nn.Module):
        def __init__(self, n_channel, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
            super().__init__()
            self.patch_to_embedding = nn.Linear(n_channel, dim)
            self.c_token = nn.Parameter(torch.randn(1, 1, dim))
            self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
            self.transformer = Transformer_block(dim, depth, heads, mlp_dim, dropout)
            self.to_c_token = nn.Identity()
            self.classifier = nn.Linear(dim, n_classes)


        def forward(self, forward_seq):
            x = self.patch_to_embedding(forward_seq)
            x = self.position(x)
            b, n, _ = x.shape
            c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
            x = torch.cat((c_tokens, x), dim=1)
            x = self.transformer(x)
            c_t = self.to_c_token(x[:, 0])
            return c_t

    class Transformer(nn.Module):
        def __init__(self, n_channels=6, len_sw=300, n_classes=11, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.3):
            super(Transformer, self).__init__()

            self.out_dim = dim
            self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            self.classifier = nn.Linear(dim, n_classes)

        def forward(self, x):
            x = self.transformer(x)
            out = self.classifier(x)
            return out
            # return out, x

    model = Transformer()
    model = model.to(device)
    print(model)

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_loss = None

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered")
                    return True
            return False
    early_stopping = EarlyStopping()

    def vote_labels(label):
        # Iterate over each sample in the batch
        votel = []
        for i in range(label.size(0)):
            # Get unique labels and their counts
            unique_labels, counts = label[i].unique(return_counts=True)

            # Find the index of the maximum count
            max_count_index = counts.argmax()

            # Get the label corresponding to that maximum count
            mode_label = unique_labels[max_count_index]

            # Append the mode to the result list
            votel.append(mode_label)

        # Convert the result list to a tensor and reshape to (batch, 1)
        vote_label = torch.tensor(votel, dtype=torch.long).view(-1)
        return vote_label

    num_epochs = 100

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    learning_rate = 0.001
    optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, amsgrad=True
            )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_losses, val_losses = [], []
    for epoch in tqdm(range(num_epochs)):
        train_loss, val_loss = [], []
        ###################
        # train the model #
        ###################
        model.train()
        true_labels, pred_labels = [], []
        for i, (sample, label) in enumerate(train_loader):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long)
            vote_label = vote_labels(label)
            vote_label = vote_label.to(device)
            output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
            loss = criterion(output, vote_label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            true_labels.append(vote_label.detach().cpu().numpy())
            pred_labels.append(output.detach().cpu().numpy())

        train_losses.append(np.average(train_loss))
        # Calculate F1 scores
        y_true = np.concatenate(true_labels, axis=0)
        y_prob = np.concatenate(pred_labels, axis=0)

        # Get the predicted class labels (argmax along the class dimension)
        y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

        # Calculate F1 score (macro F1 score)
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f'F1 Score of training set: {f1:.4f}')

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            true_labels, pred_labels = [], []
            for i, (sample, label) in enumerate(val_loader):
                sample = sample.to(device=device, dtype=torch.float)
                label = label.to(device=device, dtype=torch.long)
                vote_label = vote_labels(label)
                vote_label = vote_label.to(device)
                output = model(sample)
                loss = criterion(output, vote_label)
                val_loss.append(loss.item())
                true_labels.append(vote_label.detach().cpu().numpy())
                pred_labels.append(output.detach().cpu().numpy())
            val_losses.append(np.average(val_loss))

            # Calculate F1 scores
            y_true = np.concatenate(true_labels, axis=0)
            y_prob = np.concatenate(pred_labels, axis=0)

            # Get the predicted class labels (argmax along the class dimension)
            y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

            # Calculate F1 score (macro F1 score)
            f1 = f1_score(y_true, y_pred, average='macro')

            print(f'F1 Score of validation set: {f1:.4f}')

            # Check early stopping
            if early_stopping(np.average(val_losses)):
                print("Stopping at epoch %s." % str(epoch))
                break
        scheduler.step(np.average(val_loss))
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f'Epoch {epoch + 1}, Learning Rate: {current_lr}')
    plt.figure(figsize=(6,4))
    plt.plot(val_losses, label='valid loss')
    plt.plot(train_losses, label='train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.savefig(f'Virtual_Data_Generation/Results/{model_name}.png')

    with torch.no_grad():
        model.eval()
        true_labels, pred_labels = [], []
        for i, (sample, label) in enumerate(test_loader):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long)
            vote_label = vote_labels(label)
            # vote_label = vote_label.to(device)
            output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13

            true_labels.append(vote_label.numpy())
            pred_labels.append(output.detach().cpu().numpy())

        # Calculate F1 scores
        y_true = np.concatenate(true_labels, axis=0)
        y_prob = np.concatenate(pred_labels, axis=0)

        # Get the predicted class labels (argmax along the class dimension)
        y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

        # Calculate F1 score (macro F1 score)
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f'F1 Score of test set: {f1:.4f}')
    return f1

if __name__ == '__main__':
    model_name = 'test'
    HAR_evaluation(model_name)