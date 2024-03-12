import pandas as pd
import numpy as np
import torch
from AE import *
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt
import random
from copy import deepcopy as dc

from snp_preprocessing import *

class AEwithPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size):
        super(AEwithPredictor, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)
        self.predictor = Decoder(hidden_size, hidden_size, num_layers, output_size)
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.reconstruction_criterion = nn.MSELoss()
        self.prediction_criterion = nn.MSELoss()
        self.losses = []
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        context = h_n[-1]
        repeat_hidden = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        reconstructions = self.decoder(repeat_hidden, h_n, c_n)
        next_value_predictions = self.predictor(repeat_hidden, h_n, c_n)
        return reconstructions, next_value_predictions
    
    def learn(self, x):
        #x is a DataLoader object
        losses = []
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        # num_batches = x.shape[0] // self.batch_size
        recon_losses_for_plot = []
        pred_losses_for_plot = []
        for e in range(self.epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_pred_loss = 0
            batch_idx = 0

            for batch_idx, x_ in enumerate(x):
                x_batch = x_[:,:-1].to(device)
                y_batch = x_[:,1:].to(device)
                

                optimizer.zero_grad()
                reconstructions, next_value_predictions = self.forward(x_batch)

                recon_loss = self.reconstruction_criterion(reconstructions, x_batch)
                pred_loss = self.prediction_criterion(next_value_predictions, y_batch)
                loss = recon_loss + pred_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_pred_loss += pred_loss.item()
            
            epoch_recon_loss /= (batch_idx * (x_.shape[1]-1))
            epoch_pred_loss /= (batch_idx * (x_.shape[1]-1))
            recon_losses_for_plot.append(epoch_recon_loss)
            pred_losses_for_plot.append(epoch_pred_loss)

            scheduler.step()
            print(f'Epoch {e+1}/{self.epochs}, Loss: {loss.item()}')
            losses.append(epoch_loss / (batch_idx + 1))

        plt.plot(recon_losses_for_plot, label='Reconstruction Loss')
        plt.plot(pred_losses_for_plot, label='Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction and Prediction Losses')
        plt.legend()
        plt.savefig('recon_pred_losses.png')
        plt.show()

        self.losses = losses
        return losses

def multi_step_prediction(symbol_data, model):
    reconstruction, _ = model(symbol_data)

    half_size = symbol_data.size(1) // 2
    half_symbol_data = torch.split(symbol_data, half_size, dim=1)[0]
    half_symbol_data = half_symbol_data.to(device)
    result = torch.zeros_like(symbol_data)
    result[:, :1, :] = half_symbol_data[:, :1, :]
    first = True
    for i in range(half_size):
        with torch.no_grad():
            predictions, next_value = model(half_symbol_data)
            # first_value_of_last_half_symbol_data = half_symbol_data[:, 0, :].unsqueeze(-1)
            next_value_last_value = next_value[:, -1, :].unsqueeze(-1)
            if first:
                first = False
                result[:, 1: next_value.size(1) + 1 + i, :] = next_value
            else:
                result[:, next_value.size(1) + 1, :] = next_value_last_value
            # half_symbol_data = torch.cat((first_value_of_last_half_symbol_data, next_value), dim=1)
            half_symbol_data = next_value

            
    # print('original:', symbol_data.flatten())
    # print('predictions:', half_symbol_data.flatten())
    symbol_data = symbol_data.detach().flatten().cpu()
    half_symbol_data = result.detach().flatten().cpu()
    reconstruction = reconstruction.detach().flatten().cpu()

    #plot the predictions and the actual data
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(symbol_data, label='Actual Data')
    axs[0].plot(half_symbol_data, label='Half Symbol Data')
    axs[0].set_xlabel('Day')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Symbol Data vs Half Symbol Data')
    axs[0].legend()

    # Plot 2: symbol_data vs reconstruction
    axs[1].plot(symbol_data, label='Actual Data')
    axs[1].plot(reconstruction, label='Reconstruction')
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Symbol Data vs Reconstruction')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('multi_step_predictions_subplots_corrected.png')  # Saves the figure to file
    plt.show()



parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=20, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=2, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=1, help='Number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='Optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=1, help='Gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('-seq','--sequence_length', type=int, default=30, help='Length of the sequence')
parser.add_argument('-test','--test_on_full_data', action='store_true', help='Test the model on the full data')
args = parser.parse_args()

# Load the CSV data into a DataFrame
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

train_data, test_data, symbol_data, long_data, prices_dict = prepare_data(device, args.sequence_length)
print("Data prepared")




if args.test_on_full_data:
    model = torch.load('model.pth')
    model.eval()
    with torch.no_grad():
        i = random.randint(0, len(long_data))
        data = long_data[i].to(device).unsqueeze(0)
        predictions, next_value_predictions = model(data)

        predictions = predictions.flatten().cpu()
        data = data.flatten().cpu()

        plt.figure(figsize=(10, 6))
        plt.plot(data.detach().numpy(), label='Actual Data')
        plt.plot(predictions.detach().numpy(), label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('High Price')
        plt.title('Predictions vs Actual Data, training params:\n')
        plt.legend()
        plt.savefig('predictions_full.png')
        plt.show()
        exit()

input_size = output_size = 1
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
hidden_size = args.hidden_size
num_layers =  args.num_layers
epochs = args.epochs
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


model = AEwithPredictor(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

model.train()
model.learn(train_loader)
torch.save(model, 'model.pth')

model.eval()
random_index = torch.randint(0, len(test_data), (1,))
multi_step_prediction_data = dc(test_data[random_index])
multi_step_prediction(multi_step_prediction_data, model)

test_preds = test_data[:,1:,:].cpu()
test_data = test_data[:, :-1, :]

with torch.no_grad():
    predictions,next_value_predictions = model(test_data)

test_data = test_data.cpu()
predictions = predictions.cpu()
next_value_predictions = next_value_predictions.cpu()

loss = torch.nn.MSELoss()(next_value_predictions, test_preds)

print(f'Accuracy of predictions: {loss.item()}')


datas = [test_data[random.randint(0, len(test_data))].to(device) for _ in range(3)]
with torch.no_grad():
    for i, data in enumerate(datas):
        predictions_i, next_value_predictions_i = model(data.unsqueeze(0))

        predictions = predictions_i.flatten().cpu()
        data = data.flatten().cpu()


        plt.figure(figsize=(10, 6))
        plt.plot(data.detach().numpy(), label='Actual Data')
        plt.plot(predictions.detach().numpy(), label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('High Price')
        plt.title('Predictions vs Actual Data, training params:\n' + "hidden_size: " + str(hidden_size) + ", num_layers: " + str(num_layers) + ", epochs: " + str(epochs) + ", optimizer: " + str(optimizer) + ",\n learning_rate: " + str(learning_rate) + ", grad_clip: " + str(grad_clip) + ", batch_size: " + str(batch_size) + ", sequence_length: " + str(args.sequence_length))
        plt.legend()
        plt.savefig(f'predictions_{i}.png')
        plt.show()
        

# This setup allows you to easily use the dates for plotting and the tensors for your PyTorch network.


PLOTTING = False
if PLOTTING: 

    symbol_data = prices_dict

    googl_data = prices_dict['AMZN']  # Assuming 'GOOGL' is a key in your dictionary

    # Step 2: Separate the data into dates and prices
    dates, prices = zip(*googl_data)  # This unzips the list of tuples into two tuples

    # Step 3: Plot the data
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(dates, prices, label='AMZN Stock Price', color='blue')  # Plot dates vs. prices
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Price')  # Y-axis label
    plt.title('AMZN Stock Price Over Time')  # Title of the plot
    plt.legend()  # Add a legend
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig('AMZN_stock_price.png')  # Save the plot to a file
    plt.show()  # Display the plot
    
    googl_data = prices_dict['GOOGL']  # Assuming 'GOOGL' is a key in your dictionary

    # Step 2: Separate the data into dates and prices
    dates, prices = zip(*googl_data)  # This unzips the list of tuples into two tuples

    # Step 3: Plot the data
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(dates, prices, label='GOOGL Stock Price', color='red')  # Plot dates vs. prices
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Price')  # Y-axis label
    plt.title('GOOGL Stock Price Over Time')  # Title of the plot
    plt.legend()  # Add a legend
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig('GOOGL_stock_price.png')  # Save the plot to a file
    plt.show()  # Display the plot
    exit()
