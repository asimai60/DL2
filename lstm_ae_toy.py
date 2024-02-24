import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from AE import *
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create synthetic data
synthetic_data = torch.rand(10_000, 50, 1)
i = np.random.randint(20, 30)
for x_ in synthetic_data:
    x_[0][i-5:i+5] *= 0.1

total_size = len(synthetic_data)
train_size = int(0.6 * total_size)
validation_size = test_size = int(0.2 * total_size)

train_data, validation_data, test_data = random_split(synthetic_data, [train_size, validation_size, test_size])


#later turn this into an argparse apllication

input_size = output_size = 1

hidden_size = 2
num_layers = 5
epochs = 50
optimizer = torch.optim.Adam
learning_rate = 0.01
grad_clip = 1
batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size)
model.train(train_loader)

# Plot the loss
plt.plot(model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')

plt.show()

# Test the model
predictions = model(test_data)
print(predictions - test_data)

#plot the predictions
