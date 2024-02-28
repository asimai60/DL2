import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
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
        h_0 = h_0.to(x.device)
        c_0 = c_0.to(x.device)
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
    
    def forward(self, z, h_n, c_n):
        lstm_out, (h_n, c_n) = self.lstm(z, (h_n, c_n))
        predictions = self.fc(lstm_out)
        return predictions

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.losses = []
    
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        repeat_hidden = h_n[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        predictions = self.decoder(repeat_hidden, h_n, c_n)
        return predictions
    
    def learn(self, x):
        #x is a DataLoader object
        losses = []
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # num_batches = x.shape[0] // self.batch_size
        for e in range(self.epochs):
            epoch_loss = 0
            batch_idx = 0
            for batch_idx, x_batch in enumerate(x):
                optimizer.zero_grad()
                predictions = self.forward(x_batch)
                cur_loss = loss = self.criterion(predictions, x_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += cur_loss.item()
            # scheduler.step()
            epoch_loss /= batch_idx
            losses.append(epoch_loss)


            
            print(f'Epoch: {e+1}/{self.epochs}, Loss: {loss.item()}')       
        self.losses = losses


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    input_size = 1
    hidden_size = 8
    num_layers = 5
    output_size = 1
    epochs = 1000
    optimizer = torch.optim.Adam
    learning_rate = 0.0035
    grad_clip = 0.5
    batch_size = 64

    model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)
    x = torch.rand(100, 10, 1).to(device)
    x_loader = DataLoader(x, batch_size=batch_size, shuffle=True)

    # print(next(iter(x_loader)))
    
    model.learn(x_loader)
    with torch.no_grad():
        predictions = model(x)
        print(predictions - x)



if __name__ == '__main__':
    main()