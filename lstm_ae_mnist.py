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

mnist_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=torchvision.transforms.ToTensor())
#plot the first 10 images
fig, axs = plt.subplots(2, 5)
for i in range(10):
    axs[i//5, i%5].imshow(mnist_data[i][0][0])
    axs[i//5, i%5].set_title(mnist_data[i][1])
    axs[i//5, i%5].axis('off')
plt.show()
