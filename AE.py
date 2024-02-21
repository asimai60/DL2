import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        
        lstm_out, (h_n, c_n) = self.lstm((x.shape[0], x.shape[1], self.input_size)) # need to fix
        return h_n, c_n
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

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
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        predictions = self.decoder(x, h_n, c_n)
        return predictions
    
    def train(self, x):