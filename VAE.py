import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import torch

# ----------------------------
# Utility: set random seed
# ----------------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_value = 2025
set_random_seed(seed_value)

# ----------------------------
# Model definition (Encoder, Decoder, VAE)
# ----------------------------
feature_dim = 6   # six sensor features
latent_dim = 32

# Encoder
encoder_inputs = tf.keras.Input(shape=(None, feature_dim))  # None for variable sequence length
sequence_length_input = tf.keras.Input(shape=(1,))           # sequence length input

masked_inputs = layers.Masking(mask_value=0.)(encoder_inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(masked_inputs)
x = layers.Bidirectional(layers.LSTM(32))(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = tf.keras.Model([encoder_inputs, sequence_length_input], [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
length_input = tf.keras.Input(shape=(1,), name='length_input')
x = layers.Dense(32)(latent_inputs)
# Expand and tile to match the desired sequence length
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
# Tile the latent vector to match the sequence length for each batch
x = layers.Lambda(lambda x: tf.tile(x[0], [tf.shape(x[1])[0], tf.cast(tf.squeeze(x[1][0]), tf.int32), 1]))([x, length_input])
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(32, return_sequences=True)(x)
outputs = layers.TimeDistributed(layers.Dense(feature_dim))(x)
decoder = tf.keras.Model([latent_inputs, length_input], outputs, name='decoder')

# VAE
vae_inputs = encoder_inputs
sequence_lengths = sequence_length_input
z_mean_out, z_log_var_out, z_out = encoder([vae_inputs, sequence_lengths])
vae_outputs = decoder([z_out, sequence_lengths])
vae = tf.keras.Model([vae_inputs, sequence_lengths], vae_outputs, name='vae')

# ----------------------------
# Loss Function (再構成誤差のみ)
# ----------------------------
def reconstruction_loss_fn(y_true, y_pred):
    # マスクを利用してパディング部分は損失計算から除外する
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    # mask = tf.expand_dims(mask, axis=-1)  # expand mask to match the shape of y_true and y_pred
    masked_true = y_true * mask
    masked_pred = y_pred * mask
    rec_loss = tf.reduce_sum(tf.square(masked_true - masked_pred), axis=[1, 2])
    # マスクの総和で割る（シーケンス長に依存した正規化）
    rec_loss = rec_loss / tf.reduce_sum(mask, axis=[1, 2])
    return tf.reduce_mean(rec_loss)

# KLダイバージェンス損失をモデルに追加する
class KLLossLayer(tf.keras.layers.Layer):
    def call(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss))
        return z_mean  # We return z_mean to not break the functional API

# Use the KLLossLayer in the functional model
z_mean_out, z_log_var_out, z_out = encoder([vae_inputs, sequence_lengths])
z_mean_out = KLLossLayer()(z_mean_out, z_log_var_out)  # Add KLLossLayer here
vae_outputs = decoder([z_out, sequence_lengths])
vae = tf.keras.Model([vae_inputs, sequence_lengths], vae_outputs, name='vae')

vae.compile(optimizer='adam', loss=reconstruction_loss_fn)

# ----------------------------
# Data loading and preprocessing
# ----------------------------
sensor_columns = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                  'atr02/acc_x','atr02/acc_y','atr02/acc_z']

rootdir = r'/root/Virtual_Data_Generation'
train_summary_path = os.path.join(rootdir, "Statistics", "train_data_summary_by_op.csv")
train_summary_df = pd.read_csv(train_summary_path)
duration_label = dict(zip(train_summary_df['operation'], train_summary_df['block_size_mean'].astype(int)))

def load_operation_data(operation, required_length):
    data_dir = r'/root/Virtual_Data_Generation/data/converted'
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    sequences = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        if operation in df.iloc[:, -1].values:
            seq = df[sensor_columns].to_numpy()
            if seq.shape[0] >= required_length:
                seq = seq[:required_length, :]
            else:
                pad_amount = required_length - seq.shape[0]
                seq = np.pad(seq, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
            sequences.append(seq)
    if len(sequences) == 0:
        print(f"No data found for operation: {operation}")
        return None, None
    data = np.stack(sequences, axis=0)
    seq_lengths = np.full((data.shape[0], 1), required_length, dtype=np.int32)
    return data, seq_lengths

# ----------------------------
# Training per operation
# ----------------------------
for op, seq_len in duration_label.items():
    print(f"Training VAE for operation: {op} with sequence length: {seq_len}")
    X, seq_lengths = load_operation_data(op, seq_len)
    if X is None:
        continue
    vae.fit([X, seq_lengths], X, epochs=50, batch_size=32)

    # データの生成
    z_mean_val, _, _ = encoder.predict([X, seq_lengths])
    X_pred = decoder.predict([z_mean_val, seq_lengths])
    print(f"Generated data shape: {X_pred.shape}")
    output_dir = os.path.join(rootdir, "Generated_Data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{op}_generated.csv")
    df = pd.DataFrame(X_pred[0], columns=sensor_columns)
    df.to_csv(output_file, index=False)
