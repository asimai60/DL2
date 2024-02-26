import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from AE import *  # Ensure your AE model supports .to(device)
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create synthetic data and move it to the chosen device
synthetic_data = torch.rand(10_000, 50, 1).to(device)
i = np.random.randint(20, 30)
for x_ in synthetic_data:
    x_[0][i-5:i+5] *= 0.1

total_size = len(synthetic_data)
train_size = int(0.6 * total_size)
validation_size = test_size = int(0.2 * total_size)

train_data, validation_data, test_data = random_split(synthetic_data, [train_size, validation_size, test_size])

input_size = output_size = 1
hidden_size = 2
num_layers = 5
epochs = 10
optimizer = torch.optim.Adam
learning_rate = 0.01
grad_clip = 1
batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)

# Assuming your model's train method correctly handles device transfer inside
model.train(train_loader)

# Plot the loss
plt.plot(model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.show()

# Test the model
# Ensure the test_data is already on the correct device or move it as needed
# Here, we're assuming test_data needs to be moved to the device.
test_data_tensor = torch.stack([item for item in test_data.dataset]).to(device)
predictions = model(test_data_tensor)

# Calculate the difference
difference = predictions - test_data_tensor


#plot a sample of the predictions and the original data


# Optionally, plot the predictions if needed
