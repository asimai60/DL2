import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse




import matplotlib.pyplot as plt

class snpAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device, optimizer):
        super(snpAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.recon_decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.pred_decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.recon_linear = nn.Linear(hidden_size, input_size)
        self.pred_linear = nn.Linear(hidden_size, input_size)
        self.device = device
        self.optimizer = optimizer
        self.recon_criterion = nn.MSELoss()
        self.pred_criterion = nn.MSELoss()
        self.hidden = self.init_hidden(batch_size)
        self.to(device)



    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
    
    def forward(self, x):
        output, (hn, cn) = self.encoder(x, self.init_hidden(x.shape[0]))
        context = hn[-1]
        repeat_context = context.unsqueeze(1).repeat(1, x.size(1),1)
        reconstructed, _ = self.recon_decoder(repeat_context, (hn, cn))
        reconstructed = self.recon_linear(reconstructed)
        predicted, _ = self.pred_decoder(repeat_context, (hn, cn))
        predicted = self.pred_linear(predicted)
        return reconstructed, predicted
    

def train_model(model, train_loader,num_epochs, learning_rate=0.01, grad_clip=None):
    losses = []
    recon_losses = []
    pred_losses = []
    optimizer = model.optimizer(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_recon_losses = []
        epoch_pred_losses = []
        for i, x in enumerate(train_loader):
            ONLY_RECONSTRUCTION = False
            if ONLY_RECONSTRUCTION:
                optimizer.zero_grad()
                x = x.to(model.device)
                recon, _ = model(x)
                recon_loss = model.recon_criterion(recon, x)
                recon_loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_losses.append(recon_loss.item())
                epoch_recon_losses.append(recon_loss.item())
                continue

            x_batch = x[:, :-1, :]
            y_batch = x[:, 1:, :]


            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            
            optimizer.zero_grad()
            recon, pred = model(x_batch)
            recon_loss = model.recon_criterion(recon, x_batch)
            pred_loss = model.pred_criterion(pred, y_batch)

            loss = recon_loss + pred_loss
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_recon_losses.append(recon_loss.item())
            epoch_pred_losses.append(pred_loss.item())
        losses.append(np.mean(epoch_losses))
        recon_losses.append(np.mean(epoch_recon_losses))
        pred_losses.append(np.mean(epoch_pred_losses)) if not ONLY_RECONSTRUCTION else pred_losses.append(0)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]}')

    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(pred_losses, label='Prediction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction and Prediction Loss')
    plt.legend()
    plt.savefig('losses.png')
    plt.show()
    return losses, recon_losses, pred_losses

def min_max_scale(x):
    eps = 1e-10
    return (x - x.min()) / (x.max() - x.min() + eps)


parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=40, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=2, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=3, help='Number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='Optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=5, help='Gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=64, help='Batch size for training')
args = parser.parse_args()

input_size = output_size = 1
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
hidden_size = args.hidden_size
num_layers =  args.num_layers
epochs = args.epochs
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


subseq_len = 50


df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
df = df[['date', 'symbol', 'high']]
df.dropna(subset=['date'], inplace=True)
df = df.groupby('symbol').filter(lambda x: x['high'].count() >= 1000)
df.dropna(inplace=True)

symbols_dict = df.groupby('symbol')['high'].apply(list).to_dict()
symbols_dict = {k:torch.tensor(v).unsqueeze(-1) for k, v in symbols_dict.items()}

data = [v for k, v in symbols_dict.items()]
data = [d[:1000] for d in data if len(d) > 1000]

data_length = len(data)
train_data = data[:int(data_length*0.8)]
train_data = torch.stack(train_data)
train_data = train_data.reshape(-1, subseq_len)
train_data = [min_max_scale(d) for d in train_data]
train_data = torch.stack(train_data).unsqueeze(-1)


test_data = data[int(data_length*0.8):]


test_data = torch.stack(test_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = snpAE(input_size, hidden_size, num_layers, batch_size, device, optimizer)
model.to(device)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

losses, recon_losses, pred_losses = train_model(model, train_loader, epochs, learning_rate=learning_rate, grad_clip=grad_clip)

for i in range(3):
    random_test_index = np.random.randint(0, len(test_data)-1)
    test_sample = test_data[random_test_index:random_test_index+1]

    test_sample_subseqs = min_max_scale(test_sample[:,:50,:]).reshape(-1, subseq_len, 1).to(device)
    recon, pred = model(test_sample_subseqs)
    recon = recon.squeeze().detach().cpu().numpy()
    pred = pred.squeeze().detach().cpu().numpy()
    test_sample_subseqs = test_sample_subseqs.squeeze().detach().cpu().numpy()

    ONLY_RECONSTRUCTION = False
    if ONLY_RECONSTRUCTION:
        plt.plot(test_sample_subseqs, label='Original')
        plt.plot(recon, label='Reconstructed')
        plt.xlabel('Days')
        plt.ylabel('Normalized Price')
        plt.title('Original vs. Reconstructed Prices')
        plt.legend()
        plt.savefig(f'recon_subseq_{i}.png')
        plt.show()
        continue
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(test_sample_subseqs, label='Original')
    axs[0].plot(recon, label='Reconstructed')
    axs[0].set_xlabel('Days')
    axs[0].set_ylabel('Normalized Price')
    axs[0].set_title('Original vs. Reconstructed Prices')
    axs[0].legend()

    axs[1].plot(test_sample_subseqs[1:], label='Original')
    axs[1].plot(pred[:-1], label='Predicted')
    axs[1].set_xlabel('Days')
    axs[1].set_ylabel('Normalized Price')
    axs[1].set_title('Original vs. Predicted Prices')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'recon_pred_subseq_{i}.png')
    plt.show()




# multi step prediction:
    
def generate_predictions(model, test_data, subseq_len=50, prediction_len=25):
    random_index = np.random.randint(0, len(test_data) - 1)
    test_sample = test_data[random_index:random_index + 1]
    
    test_sample_subseqs = min_max_scale(test_sample[:, :subseq_len, :]).reshape(-1, subseq_len, 1).to(model.device)
    
    half_sample = test_sample_subseqs[:, :prediction_len, :]
    
    with torch.no_grad():
        _, initial_pred = model(half_sample)
    
    result = torch.cat([half_sample[:, :1, :], initial_pred], dim=1)
    
    # Generate predictions iteratively
    for i in range(1, prediction_len):
        with torch.no_grad():
            _, pred = model(result[:, -prediction_len:, :])
            result = torch.cat([result, pred[:, -1:, :]], dim=1) 

    result = result.squeeze().cpu().numpy()
    test_sample_subseqs = test_sample_subseqs.squeeze().cpu().numpy()

    plt.plot(test_sample_subseqs, label='Original')
    plt.plot(result, label='Predicted')
    plt.xlabel('Days')
    plt.ylabel('Normalized Price')
    plt.title('Original vs. Predicted Prices')
    plt.legend()
    plt.savefig('multi_step_pred.png')
    plt.show()

generate_predictions(model, test_data, subseq_len=100, prediction_len=50)