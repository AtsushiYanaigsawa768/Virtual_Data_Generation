import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.losses import mse
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
# Here we use only the 6 sensor columns (exclude the 'operation' column)
csv_path = 'your_data.csv'  # Adjust path as needed
df = pd.read_csv(csv_path)

selected_columns_acc = [
    'atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z',
    'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z'
]
data = df[selected_columns_acc].values

# Scale data to [0,1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# -------------------------------
# 2. VAE Hyperparameters
# -------------------------------
original_dim = data_scaled.shape[1]  # now 6 (only acceleration data)
intermediate_dim = 64
latent_dim = 2

# -------------------------------
# 3. Construct the Encoder
# -------------------------------
inputs = layers.Input(shape=(original_dim,), name='encoder_input')
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(h)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# -------------------------------
# 4. Construct the Decoder
# -------------------------------
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
# Output uses 'linear' activation for regression
outputs = layers.Dense(original_dim, activation='linear')(x)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# -------------------------------
# 5. Construct and Train the VAE
# -------------------------------
vae_outputs = decoder(z)
vae = Model(inputs, vae_outputs, name='vae_mlp')

# Define VAE loss: reconstruction loss + KL divergence
reconstruction_loss = mse(inputs, vae_outputs)
reconstruction_loss *= original_dim

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.summary()

epochs = 50
batch_size = 32
vae.fit(data_scaled, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1)

# -------------------------------
# 6. Generate New Samples
# -------------------------------
n_samples = 10
z_samples = np.random.normal(size=(n_samples, latent_dim))
generated_data_scaled = decoder.predict(z_samples)
generated_data = scaler.inverse_transform(generated_data_scaled)

gen_df = pd.DataFrame(generated_data, columns=selected_columns_acc)
print(gen_df)
