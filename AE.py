import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        #x is a DataLoader object

        (h_0, c_0) = self.init_hidden(len(x))
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return h_n, c_n
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size), 
                torch.zeros(self.num_layers, batch_size, self.hidden_size)) #(h_0, c_0), might need to change num layers to sequence length

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h_n, c_n):
        lstm_out, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        predictions = self.fc(lstm_out)
        return predictions

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, num_layers, output_size)
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.losses = []
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        predictions = self.decoder(x, h_n, c_n)
        return predictions
    
    def train(self, x):
        #x is a DataLoader object
        losses = []
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # num_batches = x.shape[0] // self.batch_size
        for e in range(self.epochs):

            for batch_idx, x_batch in enumerate(x):
                
                optimizer.zero_grad()
                predictions = self.forward(x_batch)
                loss = self.criterion(predictions, x_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                losses.append(loss.item())

            print(f'Epoch: {e+1}/{self.epochs}, Loss: {loss.item()}')
        self.losses = losses


def main():
    input_size = 10
    hidden_size = 5
    num_layers = 5
    output_size = 10
    epochs = 1000
    optimizer = torch.optim.Adam
    learning_rate = 0.01
    grad_clip = 1
    batch_size = 32

    model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size)
    x = torch.rand(100, 10, 10)
    x_loader = DataLoader(x, batch_size=batch_size, shuffle=True)

    print(next(iter(x_loader)))
    
    model.train(x_loader)
    predictions = model(x)
    print(predictions - x)



if __name__ == '__main__':
    main()