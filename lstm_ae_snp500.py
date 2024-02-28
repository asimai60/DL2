import pandas as pd
import torch
from AE import *
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt
import random
from snp_preprocessing import prepare_data

class AEwithPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size):
        super(AEwithPredictor, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)
        self.predictor = nn.Linear(hidden_size, output_size)
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
        next_value_predictions = self.predictor(context)
        return reconstructions, next_value_predictions
    
    def learn(self, x, y):
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

            for batch_idx, x_batch in enumerate(x):
                x_batch = x_batch.to(device)
                y_batch = y[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].unsqueeze(1).to(device)

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
            
            epoch_recon_loss /= batch_idx
            epoch_pred_loss /= batch_idx
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
    for i in range(half_size):
        with torch.no_grad():
            predictions, next_value = model(half_symbol_data)
        half_symbol_data = torch.cat((half_symbol_data, next_value.unsqueeze(0)), dim=1)
    # print('original:', symbol_data.flatten())
    # print('predictions:', half_symbol_data.flatten())
    symbol_data = symbol_data.flatten().cpu()
    half_symbol_data = half_symbol_data.flatten().cpu()
    reconstruction = reconstruction.flatten().cpu()

    #plot the predictions and the actual data
    plt.figure(figsize=(10, 6))
    plt.plot(symbol_data.detach().numpy(), label='Actual Data')
    plt.plot(half_symbol_data.detach().numpy(), label='Predictions')
    plt.plot(reconstruction.detach().numpy(), label='reconstruction')
    plt.xlabel('Day')
    plt.ylabel('High Price')
    plt.title('Predictions vs Actual Data')
    plt.legend()
    plt.savefig('multi_step_predictions.png')
    plt.show()



parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=20, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=2, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=5, help='Number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='Optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=1, help='Gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('-seq','--sequence_length', type=int, default=30, help='Length of the sequence')
args = parser.parse_args()

# Load the CSV data into a DataFrame
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

train_data, test_data, train_preds, test_preds , symbol_data = prepare_data(device, args.sequence_length)
print("Data prepared")

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
train_preds = train_preds.to(device)

model.train()
model.learn(train_loader, train_preds)

model.eval()
random_index = torch.randint(0, len(test_data), (1,))
multi_step_prediction_data = test_data[random_index]
multi_step_prediction(multi_step_prediction_data, model)

with torch.no_grad():
    predictions,next_value_predictions = model(test_data)


test_data = test_data.cpu()
predictions = predictions.cpu()
test_preds = test_preds.unsqueeze(1).cpu()
next_value_predictions = next_value_predictions.cpu()
distance = next_value_predictions - test_preds
#accuracy of next value predictions
loss = torch.nn.MSELoss()(next_value_predictions, test_preds)
accuracy = torch.mean(torch.abs(distance))
print(f'Accuracy of next value predictions: {accuracy.item()}')


i = random.randint(0, len(test_data))
index = len(symbol_data) - i
for key, value in symbol_data.items():
    if index > 0:
        index -= 1
        continue
    else:
        data = value[0]
        with torch.no_grad():
            predictions, next_value_predictions = model(data.unsqueeze(0))
            predictions = predictions * (value[2][1] - value[2][0]) + value[2][0]
            data = data * (value[2][1] - value[2][0]) + value[2][0]

            predictions = predictions.flatten().cpu()
            data = data.flatten().cpu()


            plt.figure(figsize=(10, 6))
            plt.plot(data.detach().numpy(), label='Actual Data')
            plt.plot(predictions.detach().numpy(), label='Predictions')
            plt.xlabel('Date')
            plt.ylabel('High Price')
            plt.title('Predictions vs Actual Data, training params:\n' + "hidden_size: " + str(hidden_size) + ", num_layers: " + str(num_layers) + ", epochs: " + str(epochs) + ", optimizer: " + str(optimizer) + ",\n learning_rate: " + str(learning_rate) + ", grad_clip: " + str(grad_clip) + ", batch_size: " + str(batch_size) + ", sequence_length: " + str(args.sequence_length))
            plt.legend()
            plt.savefig('predictions_.png')
            plt.show()
        
        break

# This setup allows you to easily use the dates for plotting and the tensors for your PyTorch network.
PLOTTING = False
if PLOTTING: 

    import matplotlib.pyplot as plt

    # Example symbol
    symbol = 'AMZN'

    # Assuming you've run the previous script and have symbol_data prepared
    amzn_dates = symbol_data[symbol]['dates']
    amzn_high_prices = symbol_data[symbol]['high_tensor'].numpy()  # Converting tensor to numpy array for plotting

    # plot AMZN high prices
    plt.figure(figsize=(10, 6))
    plt.plot(amzn_dates, amzn_high_prices, label='AMZN High Prices')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('AMZN Stock High Prices Over Time')
    plt.legend()
    plt.show()

    #plot GOOGL high prices
    symbol = 'GOOGL'
    googl_dates = symbol_data[symbol]['dates']
    googl_high_prices = symbol_data[symbol]['high_tensor'].numpy()  # Converting tensor to numpy array for plotting

    plt.figure(figsize=(10, 6))
    plt.plot(googl_dates, googl_high_prices, label='GOOGL High Prices')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('GOOGL Stock High Prices Over Time')
    plt.legend()
    plt.show()


