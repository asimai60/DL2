import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split

from AE import *     

import matplotlib.pyplot as plt

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#transform the data to a tensor and normalize it
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,))])


#mnist_data is a tensor, it holds num_images X 2, where the first element is the image and the second is the label
mnist_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=transform)


#shapes and sizes
# print(len(mnist_data)) #60000
# print(mnist_data[0][0].shape) #torch.Size([1, 28, 28])
# print(mnist_data[0][1]) #5

## plot the first 10 images
# fig, axs = plt.subplots(2, 5)
# for i in range(10):
#     axs[i//5, i%5].imshow(mnist_data[i][0][0])
#     axs[i//5, i%5].set_title(mnist_data[i][1])
#     axs[i//5, i%5].axis('off')
# plt.show()

# mnist_images = mnist_data.data.unsqueeze(1).float()
# mnist_labels = mnist_data.targets

#shapes and sizes
# print(mnist_images.shape) #torch.Size([60000, 1, 28, 28])
# print(mnist_labels.shape) #torch.Size([60000])


# Create a DataLoader
total_size = len(mnist_data)
train_size = int(0.6 * total_size)
validation_size = test_size = int(0.2 * total_size)
train_, validation_, test_ = random_split(mnist_data, [train_size, validation_size, test_size])

train_images = torch.stack([image for image, label in train_])
train_labels = [label for image, label in train_]
validation_images = torch.stack([image for image, label in validation_])
validation_labels = [label for image, label in validation_]
test_images = torch.stack([image for image, label in test_])
test_labels = [label for image, label in test_]

# print(train_images[0][0].shape)
# fig, axs = plt.subplots(2, 5)
# for i in range(10):
#     axs[i//5, i%5].imshow(train_images[i][0])
#     axs[i//5, i%5].set_title(train_labels[i])
#     axs[i//5, i%5].axis('off')
# plt.show()

#later turn this into an argparse apllication
input_size = output_size = 28
hidden_size = 27
num_layers = 2
epochs = 10
optimizer = torch.optim.Adam
learning_rate = 0.005
grad_clip = 0.5
batch_size = 64

train_images = train_images.view(-1, 28, 28)
validation_images = validation_images.view(-1, 28, 28)
test_images = test_images.view(-1, 28, 28)


train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_images, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=True)

model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size)
model.train(train_loader)

# Plot the loss
plt.plot([i+1 for i in range(epochs)], model.losses)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')

plt.show()

# Test the model
predictions = model(test_images)

print(predictions - test_images)

#plot the predictions
fig, axs = plt.subplots(2, 5)
for i in range(10):
    axs[i//5, i%5].imshow(predictions[i].detach().numpy())
    axs[i//5, i%5].set_title(test_labels[i])
    axs[i//5, i%5].axis('off')
plt.show()


