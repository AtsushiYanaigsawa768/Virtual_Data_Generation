import os
import random
import pandas as pd
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_value = 2025
set_random_seed(seed_value)

# Set paths
REAL_DATA = True
realpath = r'/data/real'
rootdir = r'/root/Virtual_Data_Generation'  # replace with your project path
real_directory = rootdir + realpath

# Specify columns
selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                    'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                    'timestamp','operation']
# For per-block calculation we only need the first six variables and operation
explanatory_vars = selected_columns[:6]
operation_col = 'operation'

# Specify training users
train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109', 'U0111', 'U0201', 'U0202', 'U0203', 'U0204', 'U0206', 'U0207', 'U0208', 'U0210']

# Build user_paths from files that end with 'S0100.csv'
user_paths = {}
for root, dirs, files in os.walk(real_directory):
    for file in files:
        if file.endswith('S0100.csv'):
            # Extract user id (assumed to be the part before the last 10 characters)
            user_id = file[:-10]
            user_paths[user_id] = os.path.join(root, file)
        else:
            os.remove(os.path.join(root, file))  # remove unused data

# Load CSV files of training users and combine them
dataframes = []
for u in train_users:
    if u in user_paths:
        df = pd.read_csv(user_paths[u], usecols=selected_columns)
        dataframes.append(df)
    else:
        print("Warning: user %s file not found." % u)

if not dataframes:
    raise ValueError("No training data found.")

# Combine all training data and sort by timestamp
all_train_data = pd.concat(dataframes, ignore_index=True)
all_train_data.sort_values(by='timestamp', inplace=True)
all_train_data.reset_index(drop=True, inplace=True)

# Process contiguous blocks (each block: contiguous rows with same operation)
block_stats = []
current_op = None
block_rows = []
for i, row in all_train_data.iterrows():
    op = row[operation_col]
    if current_op is None:
        # First row, start a new block
        current_op = op
        block_rows.append(row)
    elif op == current_op:
        block_rows.append(row)
    else:
        # Block changed, compute stats for the finished block
        block_df = pd.DataFrame(block_rows)
        temp = {
            'operation': current_op,
            'block_size': len(block_df)
        }
        # For each explanatory variable, compute block mean and variance
        for col in explanatory_vars:
            temp[f'{col}_mean'] = block_df[col].mean()
            temp[f'{col}_var'] = block_df[col].var()
        block_stats.append(temp)
        # Start new block
        current_op = op
        block_rows = [row]

# Process the last block if exists
if block_rows:
    block_df = pd.DataFrame(block_rows)
    temp = {
        'operation': current_op,
        'block_size': len(block_df)
    }
    for col in explanatory_vars:
        temp[f'{col}_mean'] = block_df[col].mean()
        temp[f'{col}_var'] = block_df[col].var()
    block_stats.append(temp)

# Convert block_stats to DataFrame
blocks_df = pd.DataFrame(block_stats)

# Group by operation and compute aggregated statistics:
# - For block sizes: mean and variance.
# - For each explanatory variable: 
#    • mean of the block means and variance of the block means,
#    • mean of the block variances and variance of the block variances.
agg_results = []
grouped = blocks_df.groupby('operation')
for op, group in grouped:
    result = {'operation': op}
    # Block size aggregated statistics
    result['block_size_mean'] = group['block_size'].mean()
    result['block_size_var'] = group['block_size'].var()
    
    for col in explanatory_vars:
        # Calculate aggregated mean of block means and its variance
        col_mean_values = group[f'{col}_mean']
        result[f'{col}_block_mean'] = col_mean_values.mean()
        result[f'{col}_block_var'] = col_mean_values.var()

    
    agg_results.append(result)

agg_df = pd.DataFrame(agg_results)

# Save aggregated summary to CSV
output_path = os.path.join(rootdir, 'train_data_summary_by_op.csv')
agg_df.to_csv(output_path, index=False)
print("Aggregated summary CSV file created at:", output_path)