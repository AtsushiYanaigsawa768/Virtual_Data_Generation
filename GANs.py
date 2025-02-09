"""
cGANs for Time-Series Data Generation using LSTM
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import time

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
num_epochs  = 1000  # Adjust as needed

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
    # Train Discriminator
    noise = tf.random.normal([batch_size, noise_dim])
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
    noise = tf.random.normal([batch_size, noise_dim])
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
    # Example: collect real CSV files into a dictionary (modify as needed)
    selected_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                        'atr02/acc_x','atr02/acc_y','atr02/acc_z',
                        'timestamp','operation']
    train_data_dict = {}
    train_users = ['U0102', 'U0103', 'U0104', 'U0106', 'U0109', 'U0111',
                   'U0201', 'U0202', 'U0203', 'U0204', 'U0206', 'U0207',
                   'U0208', 'U0210']
    
    # Walk through real_directory and read CSV files named with 'S0100.csv'
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

    print(user_paths)
    for u in train_users:
        if u in user_paths:
            # Read only the selected columns
            df = pd.read_csv(user_paths[u], usecols=selected_columns)
            # Note: adjust reshaping strategy based on how your real CSV is structured.
            # Here we group every seq_len consecutive rows into one sample.
            data = df[selected_columns[:6]].values.astype(np.float32)
            label = df[selected_columns[-1]].values.astype(np.float32)
            n_samples = data.shape[0] // seq_len
            data = data[:n_samples*seq_len].reshape(n_samples, seq_len, feature_dim)
            label = label[:n_samples*seq_len].reshape(n_samples, seq_len)[:, 0].reshape(n_samples, 1)
            train_data_dict[u] = (data, label)
        else:
            print(f"Warning: No CSV file found for user {u}. Skipping.")
    
    # Combine all users' data
    all_data = np.concatenate([d for (d, l) in train_data_dict.values()], axis=0)
    all_labels = np.concatenate([l for (d, l) in train_data_dict.values()], axis=0)
    
else:
    # Create dummy data: 1000 samples of time-series with seq_len time steps
    num_samples = 1000
    all_data = np.random.uniform(-1, 1, size=(num_samples, seq_len, feature_dim)).astype(np.float32)
    # Dummy labels: in this example, use a constant value (e.g., 8100.0) for all samples
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
    for u, (real_ts, real_labels) in train_data_dict.items():
        num_samples = real_labels.shape[0]
        virtual_ts = generate_virtual_data(real_labels, num_samples)
        save_virtual_data(virtual_ts, real_labels, u)
else:
    # For dummy data, generate virtual data and save as one file.
    num_samples = all_labels.shape[0]
    virtual_ts = generate_virtual_data(all_labels, num_samples)
    save_virtual_data(virtual_ts, all_labels, "virtual_dummy")

# Optionally, further evaluation function can be called here.