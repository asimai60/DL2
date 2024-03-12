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

parser = argparse.ArgumentParser(description='Train an autoencoder on synthetic data')
parser.add_argument('--hidden_size', type=int, default=10, help='The hidden size of the LSTM')
parser.add_argument('--num_layers', type=int, default=2, help='The number of layers in the LSTM')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train the model')
parser.add_argument('--optimizer', type=str, default='Adam', help='The optimizer to use')
parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate for the optimizer')
parser.add_argument('--grad_clip', type=int, default=1, help='The gradient clipping value')
parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training')
parser.add_argument('--random', action='store_true' , help='Whether to use a random seed')



args = parser.parse_args()


input_size = output_size = 1
hidden_size = args.hidden_size
num_layers = args.num_layers
epochs = args.epochs
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


# Set the random seed for reproducibility
if not args.random:    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

# Create synthetic data and move it to the chosen device
synthetic_data = torch.rand(10_000, 50, 1).to(device)
i = np.random.randint(20, 30)
for j, x_ in enumerate(synthetic_data):
    mask = torch.tensor([1]*(i-5) + [0.1]*10 + [1]*(45-i))
    synthetic_data[j] = (x_.flatten() * mask).reshape(50, 1)
    

total_size = len(synthetic_data)
train_size = int(0.6 * total_size)
validation_size = test_size = int(0.2 * total_size)

random_index = np.random.randint(0, len(synthetic_data), (1,))
random_index2 = np.random.randint(0, len(synthetic_data), (1,))
data_sample = synthetic_data[random_index]
data_sample2 = synthetic_data[random_index2]

fig, axs = plt.subplots(2, figsize=(10, 6))  # Set a figure size for better visibility

# Plotting the first sample
axs[0].plot(data_sample.detach().cpu().numpy().flatten())
axs[0].set_xlabel('Time')  # Set x-axis label for the first subplot
axs[0].set_ylabel('Value')  # Set y-axis label for the first subplot

# Plotting the second sample
axs[1].plot(data_sample2.detach().cpu().numpy().flatten())
axs[1].set_xlabel('Time')  # Set x-axis label for the second subplot
axs[1].set_ylabel('Value')  # Set y-axis label for the second subplot

# Setting the title for the whole figure at the top
fig.suptitle('Synthetic Data Example', fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the suptitle

plt.savefig('synthetic_data_example.png')  # Save the figure as a PNG file
plt.show()  # Display the plot





train_data, validation_data, test_data = random_split(synthetic_data, [train_size, validation_size, test_size])







train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)

# Assuming your model's train method correctly handles device transfer inside
model.train()
model.learn(train_loader)

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
