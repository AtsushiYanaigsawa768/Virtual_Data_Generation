import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Hyperparameters
# -------------------------------
seq_len = 10         # Length of each time-series sample
batch_size = 32      # Batch size
noise_dim = 16       # Dimensionality of noise (latent vector)
condition_dim = 1    # Dimensionality of condition (e.g. operation value)
sensor_dim = 6       # Number of sensor channels (e.g., atr01/acc_x ~ atr02/acc_z)
hidden_dim = 32      # LSTM hidden dimension
num_layers = 2       # Number of LSTM layers
num_epochs = 1000    # Number of epochs
lr = 0.001           # Learning rate
REAL_DATA = False    # Set to True to load CSV data instead of using dummy samples

# -------------------------------
# Generator Definition
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, hidden_dim, sensor_dim, num_layers):
        super(Generator, self).__init__()
        # Input will be the concatenation of noise and condition
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.input_dim = noise_dim + condition_dim
        
        # LSTM to process time-series information
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, sensor_dim)
        self.relu = nn.ReLU()
        
    def forward(self, noise, condition):
        """
        noise: (batch, seq_len, noise_dim)
        condition: (batch, condition_dim) -> condition repeated for each timestep
        """
        # Repeat condition to match sequence length: (batch, seq_len, condition_dim)
        condition_expanded = condition.unsqueeze(1).repeat(1, noise.size(1), 1)
        # Concatenate noise and condition
        gen_input = torch.cat([noise, condition_expanded], dim=2)
        out, _ = self.lstm(gen_input)
        out = self.relu(out)
        # Produce sensor data at each timestep
        out = self.fc(out)
        return out

# -------------------------------
# Discriminator Definition
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, sensor_dim, condition_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        # Input is the sensor data concatenated with condition
        self.sensor_dim = sensor_dim
        self.condition_dim = condition_dim
        self.input_dim = sensor_dim + condition_dim
        
        # LSTM to extract features; use the last timestep output for classification
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, condition):
        """
        x: (batch, seq_len, sensor_dim)
        condition: (batch, condition_dim)
        """
        condition_expanded = condition.unsqueeze(1).repeat(1, x.size(1), 1)
        disc_input = torch.cat([x, condition_expanded], dim=2)
        out, _ = self.lstm(disc_input)
        # Use last timestep for classification
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# -------------------------------
# Data preparation functions
# -------------------------------
def generate_real_samples(num_samples, seq_len, sensor_dim, operation_value=8100.0):
    """
    Creates dummy time-series data: a sine wave plus noise (as an example sensor signal).
    Returns data of shape (num_samples, seq_len, sensor_dim) and a condition array of shape (num_samples, 1).
    """
    x_vals = np.linspace(0, 10, seq_len)
    data = []
    for _ in range(num_samples):
        sequence = []
        for _ in range(sensor_dim):
            sequence.append(np.sin(x_vals + np.random.rand() * 2 * np.pi) + 
                            np.random.normal(0, 0.1, seq_len))
        sequence = np.stack(sequence, axis=1)  # shape: (seq_len, sensor_dim)
        data.append(sequence)
    data = np.array(data).astype(np.float32)
    condition = np.ones((num_samples, 1), dtype=np.float32) * operation_value
    return data, condition

def load_real_data(csv_dir, seq_len, sensor_cols, label_col):
    """
    Loads real CSV files from csv_dir.
    csv_dir: directory containing CSV files
    sensor_cols: list of sensor columns
    label_col: column name for condition
    Returns:
       data: np.array of shape (n_samples, seq_len, sensor_dim)
       condition: np.array of shape (n_samples, 1)
    Note: This is a simple example. You may need to adjust it to match your CSV format.
    """
    data_list = []
    label_list = []
    for root, _, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, file), usecols=sensor_cols + [label_col])
                n_samples = len(df) // seq_len
                if n_samples < 1:
                    continue
                data_arr = df[sensor_cols].values.astype(np.float32)[:n_samples * seq_len]
                data_arr = data_arr.reshape(n_samples, seq_len, -1)
                labels_arr = df[label_col].values[:n_samples].reshape(n_samples, 1).astype(np.float32)
                data_list.append(data_arr)
                label_list.append(labels_arr)
    if not data_list:
        raise FileNotFoundError("No CSV files found in " + csv_dir)
    data = np.concatenate(data_list, axis=0)
    condition = np.concatenate(label_list, axis=0)
    return data, condition

# -------------------------------
# Set device and initialize models
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(noise_dim, condition_dim, hidden_dim, sensor_dim, num_layers).to(device)
discriminator = Discriminator(sensor_dim, condition_dim, hidden_dim, num_layers).to(device)

criterion = nn.BCELoss()  # Binary cross entropy loss
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# -------------------------------
# Data Loading
# -------------------------------
if REAL_DATA:
    # Example: load from CSV directory (adjust directory and column names as needed)
    csv_directory = r'/root/Virtual_Data_Generation/data/converted'
    sensor_cols = ['atr01/acc_x','atr01/acc_y','atr01/acc_z',
                   'atr02/acc_x','atr02/acc_y','atr02/acc_z']
    label_col = 'operation'
    real_data_np, real_condition_np = load_real_data(csv_directory, seq_len, sensor_cols, label_col)
else:
    num_samples = 1000
    real_data_np, real_condition_np = generate_real_samples(num_samples, seq_len, sensor_dim)

# Convert numpy arrays to torch tensors
real_data_tensor = torch.tensor(real_data_np).to(device)
real_condition_tensor = torch.tensor(real_condition_np).to(device)
total_samples = real_data_tensor.size(0)

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(num_epochs):
    # --- Train Discriminator ---
    discriminator.zero_grad()
    # Select a random batch of real samples
    idx = np.random.randint(0, total_samples, batch_size)
    real_seq = real_data_tensor[idx]         # (batch, seq_len, sensor_dim)
    real_cond = real_condition_tensor[idx]     # (batch, condition_dim)
    real_labels = torch.ones(batch_size, 1).to(device)
    
    output_real = discriminator(real_seq, real_cond)
    loss_real = criterion(output_real, real_labels)

    # Generate fake samples with generator, using same condition as real samples
    noise = torch.randn(batch_size, seq_len, noise_dim).to(device)
    fake_seq = generator(noise, real_cond)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    output_fake = discriminator(fake_seq.detach(), real_cond)
    loss_fake = criterion(output_fake, fake_labels)
    
    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()
    
    # --- Train Generator ---
    generator.zero_grad()
    noise = torch.randn(batch_size, seq_len, noise_dim).to(device)
    fake_seq = generator(noise, real_cond)
    # Generator aims to have discriminator label its outputs as real
    output = discriminator(fake_seq, real_cond)
    loss_G = criterion(output, real_labels)
    loss_G.backward()
    optimizer_G.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

# -------------------------------
# Generate and Save Virtual Data
# -------------------------------
def generate_virtual_data(condition_value, num_samples_to_generate):
    generator.eval()
    with torch.no_grad():
        # Create condition tensor for all samples; shape: (num_samples, 1)
        cond = torch.ones(num_samples_to_generate, 1).to(device) * condition_value
        noise = torch.randn(num_samples_to_generate, seq_len, noise_dim).to(device)
        virtual_data = generator(noise, cond)
    return virtual_data.cpu().numpy(), cond.cpu().numpy()

def save_virtual_data(data, condition, filename, virt_directory=r'/data/virtual'):
    # Here, flatten each sample to a single row (seq_len * sensor_dim)
    flattened = data.reshape(data.shape[0], seq_len * sensor_dim)
    df = pd.DataFrame(flattened)
    df["operation"] = condition
    if not os.path.exists(virt_directory):
        os.makedirs(virt_directory)
    filepath = os.path.join(virt_directory, filename + '.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved virtual data to {filepath}")

# Example: generate virtual data for operation value 8100.0
virtual_data, virtual_condition = generate_virtual_data(8100.0, total_samples)
save_virtual_data(virtual_data, virtual_condition, "virtual_data")
