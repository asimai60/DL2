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

        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self.classifier_criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        context = h_n[-1]
        repeat_hidden = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        predictions = self.decoder(repeat_hidden, h_n, c_n)
        classifier_predictions = self.classifier(context)


        if x.shape[1] == 28:
            return predictions, classifier_predictions
        else:
            return classifier_predictions
    
    def learn(self, x, y, test_x, test_y, pixel_input=False):
        losses = []
        accuracies = []
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            batch_idx = 0
            for batch_idx, x_batch in enumerate(x):
                x_batch = x_batch.to(device)
                y_batch = y[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

                optimizer.zero_grad()
                if not pixel_input:
                    predictions, classifier_predictions = self.forward(x_batch)
                    recon_loss = self.criterion(predictions, x_batch)
                else:
                    classifier_predictions = self.forward(x_batch)

                class_loss = self.classifier_criterion(classifier_predictions, y_batch)
                loss = recon_loss + class_loss if not pixel_input else class_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()

            PLOT = True
            if PLOT:
                _ ,test_class = self.forward(test_x)
                preds = torch.argmax(test_class, dim=1)
                correct = (preds == test_y).sum().item()
                accuracy = correct/len(test_y)
                accuracies.append(accuracy)
            losses.append(loss.item())
            print(f'Epoch: {epoch+1}/{self.epochs}, Loss: {loss.item()}')

        self.losses = losses

        plt.plot([i+1 for i in range(self.epochs)], losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training Loss')
        plt.savefig('recon_class_loss.png')
        plt.show()

        if PLOT:
            plt.plot([i+1 for i in range(self.epochs)], accuracies)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy')
            plt.savefig('class_accuracy.png')
            plt.show()


        


parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=27, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=2, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=5, help='Number of epochs to train the model')
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

#transform the data to a tensor and normalize it
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,))])

if args.pixel_input:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,)),
                                torchvision.transforms.Lambda(lambda x: torch.flatten(x).unsqueeze(-1))])


#mnist_data is a tensor, it holds num_images X 2, where the first element is the image and the second is the label
mnist_train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=transform)
mnist_test_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)

# Create a DataLoader

train_images = []
train_labels = []
for image, label in mnist_train_data:
    train_images.append(image)
    train_labels.append(label)
train_images = torch.stack(train_images).squeeze(1).to(device)
train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=False)
train_labels_loader = torch.tensor(train_labels).to(device)

test_images = []
test_labels = []
for image, label in mnist_test_data:
    test_images.append(image)
    test_labels.append(label)
test_images = torch.stack(test_images).squeeze(1).to(device)
test_labels_loader = torch.tensor(test_labels).to(device)



model = AeWithClassifier(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size, 10).to(device)
print("start training....")

model.train()
model.learn(train_loader, train_labels_loader,test_images,test_labels_loader,pixel_input=args.pixel_input)

# Plot the loss
plt.plot([i+1 for i in range(epochs)], model.losses)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.savefig('loss.png')
plt.show()

# Test the model
model.eval()
with torch.no_grad():
    if args.pixel_input:
        _, classifier_predictions = model(test_images)
    else:
        predictions, classifier_predictions = model(test_images)

if not args.pixel_input:
    predictions = predictions.to('cpu')
classifier_predictions = classifier_predictions.to('cpu')
#accuracy of the classifier_predictions and test_labels
correct = 0
for i in range(len(classifier_predictions)):
    if torch.argmax(classifier_predictions[i]) == test_labels[i]:
        correct += 1
print(f'Accuracy: {correct/len(classifier_predictions)}')

import random
if not args.pixel_input:
    # Adjust the subplot structure to have 3 rows and 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    
    # Select three random samples instead of ten
    sample_indices = [random.randint(0, len(predictions) - 1) for _ in range(3)]
    
    for j, i in enumerate(sample_indices):
        # Display the original image on the left column
        axs[j, 0].imshow(test_images[i].view(28, 28).detach().cpu().numpy())
        axs[j, 0].set_title(f"Original")
        axs[j, 0].axis('off')
        
        # Display the reconstructed image on the right column
        axs[j, 1].imshow(predictions[i].view(28, 28).detach().cpu().numpy())
        axs[j, 1].set_title(f"Reconstructed - {np.argmax(classifier_predictions[i].detach().cpu().numpy())}")
        axs[j, 1].axis('off')
    
    plt.savefig('comparison_original_reconstructed.png')
    plt.show()


