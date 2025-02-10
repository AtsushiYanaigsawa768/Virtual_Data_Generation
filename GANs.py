"""
cGANs for Time-Series Data Generation using LSTM
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, Input
import time
import torch
# from torch import nn
# ----------------------------
# Utility: set random seed
# ----------------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_value = 2025
set_random_seed(seed_value)

# ----------------------------
# Paths & data settings
# ----------------------------
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
REAL_DATA = True  # Set to False to use dummy data
rootdir = r'/root/Virtual_Data_Generation'  # adjust as needed
realpath = r'/data/real'
virtpath = r'/data/virtual'
real_directory = os.path.join(rootdir, realpath)
virt_directory = os.path.join(rootdir, virtpath)

# ----------------------------
# GAN hyperparameters
# ----------------------------
noise_dim   = 100   # Dimension of the noise vector
label_dim   = 1     # Label dimension (e.g., operation)
seq_len     = 6     # Number of timesteps (e.g., 6 time steps)
feature_dim = 6     # Features per time step (e.g., 6 sensor channels)
batch_size  = 32
learning_rate = 0.0002
num_epochs  = 1 # Adjust as needed

# ----------------------------
# Generator definition with LSTM
# ----------------------------
def build_generator(noise_dim, label_dim, seq_len, feature_dim):
    # Noise and label inputs
    noise_input = Input(shape=(noise_dim,), name="noise_input")
    label_input = Input(shape=(label_dim,), name="label_input")
    
    # Embed the label to a smaller dimension
    label_embedding = layers.Dense(16, activation='relu')(label_input)
    
    # Concatenate noise and label embedding
    combined = layers.Concatenate()([noise_input, label_embedding])
    
    # Map to an intermediate representation and reshape to sequence format
    x = layers.Dense(seq_len * 64, activation='relu')(combined)
    x = layers.Reshape((seq_len, 64))(x)
    
    # Use LSTM layers to generate a time series
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    
    # Generate feature values at each timestep (using tanh to restrict to [-1, 1])
    output = layers.TimeDistributed(layers.Dense(feature_dim, activation='tanh'))(x)
    
    model = Model(inputs=[noise_input, label_input], outputs=output, name="Generator")
    return model

# ----------------------------
# Discriminator definition with LSTM
# ----------------------------
def build_discriminator(seq_len, feature_dim, label_dim):
    ts_input = Input(shape=(seq_len, feature_dim), name="timeseries_input")
    label_input = Input(shape=(label_dim,), name="label_input")
    
    # Embed and repeat the label so each timestep gets the information
    label_embedding = layers.Dense(16, activation='relu')(label_input)
    repeated_label = layers.RepeatVector(seq_len)(label_embedding)
    
    # Concatenate timeseries and label at each timestep
    combined = layers.Concatenate(axis=-1)([ts_input, repeated_label])
    
    # LSTM layers to extract temporal features
    x = layers.LSTM(64, return_sequences=True)(combined)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Classification: real (1) or fake (0)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[ts_input, label_input], outputs=output, name="Discriminator")
    return model

# ----------------------------
# Instantiate the models and optimizers
# ----------------------------
generator = build_generator(noise_dim, label_dim, seq_len, feature_dim)
discriminator = build_discriminator(seq_len, feature_dim, label_dim)

g_optimizer = tf.keras.optimizers.Adam(learning_rate)
d_optimizer = tf.keras.optimizers.Adam(learning_rate)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# ----------------------------
# Training step (one batch)
# ----------------------------
@tf.function
def train_step(real_ts, real_labels):
    # Get current batch size dynamically
    current_batch_size = tf.shape(real_labels)[0]
    # Train Discriminator
    noise = tf.random.normal([current_batch_size, noise_dim])
    with tf.GradientTape() as d_tape:
        fake_ts = generator([noise, real_labels], training=True)
        real_output = discriminator([real_ts, real_labels], training=True)
        fake_output = discriminator([fake_ts, real_labels], training=True)
        
        d_loss_real = bce_loss(tf.ones_like(real_output), real_output)
        d_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    # Train Generator
    noise = tf.random.normal([current_batch_size, noise_dim])
    with tf.GradientTape() as g_tape:
        fake_ts = generator([noise, real_labels], training=True)
        fake_output = discriminator([fake_ts, real_labels], training=True)
        g_loss = bce_loss(tf.ones_like(fake_output), fake_output)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, g_loss

# ----------------------------
# Data preparation
# ----------------------------
def get_tf_dataset(timeseries, labels):
    dataset = tf.data.Dataset.from_tensor_slices((timeseries, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset

if REAL_DATA:
    # Use csv files in /data/conveted with "train" in the filename
    converted_dir = rootdir + r'/data/converted'
    train_files = []
    for root, dirs, files in os.walk(converted_dir):
        for file in files:
            if any(user in file for user in train_users)  and file.endswith('.csv'):
                train_files.append(os.path.join(root, file))
                
    if not train_files:
        raise FileNotFoundError("No csv files with 'train' found in /data/conveted.")
    
    train_data_list = []
    for csv_file in train_files:
        print(f"Reading {csv_file}")
        # Read only the selected columns: sensor data and operation label
        df = pd.read_csv(csv_file, usecols=new_columns)
        n_samples = len(df) // seq_len
        if n_samples < 1:
            continue
        # Use only the sensor columns for timeseries data
        sensor_cols = new_columns[:-1]
        data = df[sensor_cols].values.astype(np.float32)
        data = data[:n_samples * seq_len].reshape(n_samples, seq_len, -1)
        # Extract labels from the 'operation' column
        labels = df['operation'].values[:n_samples].reshape(n_samples, 1)
        train_data_list.append((data, labels))
    # Combine data from all files
    all_data = np.concatenate([item[0] for item in train_data_list], axis=0)
    all_labels = np.concatenate([item[1] for item in train_data_list], axis=0)
else:
    # Create dummy data
    num_samples = 1000
    all_data = np.random.uniform(-1, 1, size=(num_samples, seq_len, feature_dim)).astype(np.float32)
    all_labels = np.full((num_samples, 1), 8100.0, dtype=np.float32)


dataset = get_tf_dataset(all_data, all_labels)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(num_epochs):
    start = time.time()
    d_loss_epoch = 0
    g_loss_epoch = 0
    steps = 0
    for real_ts, real_labels in dataset:
        d_loss, g_loss = train_step(real_ts, real_labels)
        d_loss_epoch += d_loss
        g_loss_epoch += g_loss
        steps += 1
    end = time.time()
    print(f"Epoch {epoch+1}/{num_epochs} - d_loss: {d_loss_epoch/steps:.4f}, "
          f"g_loss: {g_loss_epoch/steps:.4f} - {end-start:.2f} sec")

# ----------------------------
# Virtual data generation & saving
# ----------------------------
def generate_virtual_data(real_labels, num_samples):
    noise = tf.random.normal([num_samples, noise_dim])
    virtual_data = generator([noise, real_labels], training=False)
    return virtual_data.numpy()

def save_virtual_data(data, labels, filename):
    # Flatten time-series into a 2D table (one sample per row)
    # Here we average features across time or flatten; adjust as needed.
    # In this example, we flatten the sequence: new shape = (samples, seq_len*feature_dim)
    flattened = data.reshape(data.shape[0], seq_len*feature_dim)
    # Append label column
    df = pd.DataFrame(flattened)
    df["operation"] = labels
    filepath = os.path.join(virt_directory, filename + '.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved virtual data to {filepath}")

# For each user (or once globally), generate virtual data and save
if REAL_DATA:
    for u, df in train_data_dict.items():
        sensor_cols = new_columns[:-1]
        data_array = df[sensor_cols].values.astype(np.float32)
        n = (data_array.shape[0] // seq_len) * seq_len
        real_ts = data_array[:n].reshape(-1, seq_len, feature_dim)
        real_labels = df['operation'].values.reshape(-1, 1)
        num_samples = real_ts.shape[0]
        virtual_ts = generate_virtual_data(real_labels, num_samples)
        save_virtual_data(virtual_ts, real_labels, u)
    num_samples = all_labels.shape[0]
    virtual_ts = generate_virtual_data(all_labels, num_samples)
    save_virtual_data(virtual_ts, all_labels, "virtual_dummy")

# Optionally, further evaluation function can be called here.