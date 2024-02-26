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

class AeWithClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size, num_classes):
        super(AeWithClassifier, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.losses = []

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes))
        self.classifier_criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        repeat_hidden = h_n[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        predictions = self.decoder(repeat_hidden, h_n, c_n)
        context = h_n[-1]
        classifier_predictions = self.classifier(context)
        return predictions, classifier_predictions
    
    def train(self, x, y):
        losses = []
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            batch_idx = 0
            for batch_idx, x_batch in enumerate(x):
                x_batch = x_batch.to(device)
                y_batch = y[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

                optimizer.zero_grad()
                predictions, classifier_predictions = self.forward(x_batch)

                recon_loss = self.criterion(predictions, x_batch)
                class_loss = self.classifier_criterion(classifier_predictions, y_batch)
                loss = recon_loss + class_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
            losses.append(loss.item())
            print(f'Epoch: {epoch+1}/{self.epochs}, Loss: {loss.item()}')
        self.losses = losses
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=27, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=4, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='Optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=1, help='Gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('-r','--random', action='store_true' , help='Whether to use a random seed')
parser.add_argument('-pxl','--pixel_input', action='store_true', help='Whether to use pixel input or row input')
args = parser.parse_args()

input_size = output_size = 28 if not args.pixel_input else 1
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
hidden_size = args.hidden_size
num_layers =  args.num_layers
epochs = args.epochs
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


# Set the random seed for reproducibility
if not args.random:    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

#transform the data to a tensor and normalize it
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,))])

if args.pixel_input:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,)),
                                torchvision.transforms.Lambda(lambda x: torch.flatten(x))])


#mnist_data is a tensor, it holds num_images X 2, where the first element is the image and the second is the label
mnist_train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=transform)
mnist_test_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)

# train_data_loader = DataLoader(mnist_train_data, batch_size=64, shuffle=True)

# Create a DataLoader

train_images = torch.stack([image for image, label in mnist_train_data]).to(device)
train_labels = [label for image, label in mnist_train_data]
test_images = torch.stack([image for image, label in mnist_test_data]).to(device)
test_labels = [label for image, label in mnist_test_data]

#later turn this into an argparse apllication


if not args.pixel_input:
    train_images = train_images.view(-1, 28, 28)
    test_images = test_images.view(-1, 28, 28)
else:
    train_images = train_images.view(-1, 784, 1)
    test_images = test_images.view(-1, 784, 1)


train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=False)
train_labels_loader = torch.tensor(train_labels).to(device)
test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)
test_labels_loader = torch.tensor(test_labels).to(device)



model = AeWithClassifier(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size, 10).to(device)
print("start training....")

model.train(train_loader, train_labels_loader)

# Plot the loss
plt.plot([i+1 for i in range(epochs)], model.losses)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.savefig('loss.png')
plt.show()

# Test the model
with torch.no_grad():
    predictions, classifier_predictions = model(test_images)

predictions = predictions.to('cpu')
classifier_predictions = classifier_predictions.to('cpu')
#accuracy of the classifier_predictions and test_labels
correct = 0
for i in range(len(classifier_predictions)):
    if torch.argmax(classifier_predictions[i]) == test_labels[i]:
        correct += 1
print(f'Accuracy: {correct/len(classifier_predictions)}')

import random
fig, axs = plt.subplots(2, 5)
sample_indices = [random.randint(0, len(predictions)) for i in range(10)]
for j,i in enumerate(sample_indices):
    axs[j//5, j%5].imshow(predictions[i].view(28,28).detach().numpy())
    axs[j//5, j%5].set_title(np.argmax(classifier_predictions[i].detach().numpy()))
    axs[j//5, j%5].axis('off')
plt.savefig('reconstructed_images.png')
plt.show()


