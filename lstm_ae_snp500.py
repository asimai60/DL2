import pandas as pd
import torch
from AE import *
from torch.utils.data import DataLoader, TensorDataset
import argparse

parser = argparse.ArgumentParser(description='Train an autoencoder with a classifier on MNIST')
parser.add_argument('-hs', '--hidden_size', type=int, default=20, help='Size of the hidden layer')
parser.add_argument('-layers','--num_layers', type=int, default=3, help='Number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='Optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=1, help='Gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('-seq','--sequence_length', type=int, default=30, help='Length of the sequence')
args = parser.parse_args()

# Load the CSV data into a DataFrame
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

def prepare_data(device, sequence_length=30):
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')

    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date'])

    df = df[['symbol', 'date', 'high']]

    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
    sample_size = sequence_length  # e.g., 30 for monthly data
    # Initialize a dictionary to store processed data for each symbol
    symbol_data = {}
    num_subsequences = len(full_date_range) // sample_size

    all_subsequences = []  # List to store all subsequences


    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        # Reindex the DataFrame to ensure it covers the full date range
        symbol_df.set_index('date', inplace=True)
        symbol_df = symbol_df.reindex(full_date_range)
        
        # Rename the index to 'date' after reindexing
        symbol_df.index.name = 'date'
        
        # Use bfill() or ffill() to fill missing values after reindexing
        symbol_df = symbol_df.bfill().ffill()  # First backward fill, then forward fill to cover all gaps
        
        
        # Normalize the 'high' column using Min-Max scaling
        min_val = symbol_df['high'].min()
        max_val = symbol_df['high'].max()
        symbol_df['high'] = (symbol_df['high'] - min_val) / (max_val - min_val)
        
        full_sequence_high_tensor = torch.tensor(symbol_df['high'].values, dtype=torch.float)
        sequence_length = full_sequence_high_tensor.size(0)
        subsequence_length = sequence_length // num_subsequences
        trimmed_length = subsequence_length * num_subsequences
        full_sequence_high_tensor = full_sequence_high_tensor[:trimmed_length]

        subsequences = full_sequence_high_tensor.split(subsequence_length)
        all_subsequences.extend(subsequences)

    data = torch.stack(all_subsequences).unsqueeze(-1).to(device)
    dic_keys = [(symbol,i) for symbol in df['symbol'].unique() for i in range(num_subsequences)]

    data_dict = {key : data[i] for i, key in enumerate(dic_keys)}


    split_ratio = 0.8
    train_data = data[:int(split_ratio * data.size(0))]
    test_data = data[int(split_ratio * data.size(0)):]
    return train_data, test_data, data_dict

train_data, test_data, symbol_data = prepare_data(device, args.sequence_length)

initial_train_size = 365  # e.g., 365 days for daily data
step_size = 30  # e.g., 30 days for monthly steps
test_size = 30  # size of the test set

# Calculate the number of steps
total_size = train_data.size(1)
num_steps = (total_size - initial_train_size) // (step_size + test_size)



input_size = output_size = 1
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
hidden_size = args.hidden_size
num_layers =  args.num_layers
epochs = args.epochs
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

model.train()
model.learn(train_loader)

model.eval()
with torch.no_grad():
    predictions = model(test_data)

test_data = test_data.cpu()
predictions = predictions.cpu()

#plot the predictions and the actual data

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(test_data[10].detach().numpy(), label='Actual Data')
plt.plot(predictions[10].detach().numpy(), label='Predictions')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.title('Predictions vs Actual Data')
plt.legend()
plt.savefig('predictions_.png')
plt.show()


exit()

for step in range(num_steps):
    # Calculate the start and end indices for training and test sets
    train_start = 0
    train_end = initial_train_size + step * (step_size + test_size)
    test_start = train_end
    test_end = test_start + test_size

    # Check if we have enough data left for this step; if not, break
    if test_end > total_size:
        break

    # Create training and test sets for this step
    step_train_data = train_data[:][train_start:train_end].to(device)
    step_test_data = train_data[:][test_start:test_end].to(device)

    # Convert to DataLoader
    step_train_loader = DataLoader(step_train_data, batch_size=32, shuffle=False)
    step_test_loader = DataLoader(step_test_data, batch_size=32)

    # Train the model
    model.train()
    model.learn(step_train_loader)

    # Test the model
    model.eval()
    with torch.no_grad():
        predictions = model(step_test_data)

    #plot the predictions and the actual data
    stock = 'AAPL'
    stock_index = list(symbol_data.keys()).index(stock)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(step_test_data[:, stock_index, 0].cpu(), label='Actual Data')
    plt.plot(predictions[:, stock_index, 0].cpu(), label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('Predictions vs Actual Data')
    plt.legend()
    plt.savefig(f'predictions_{stock}.png')
    plt.show()

    # Calculate the loss

    print(f'Step {step + 1}/{num_steps}, Loss: {model.criterion(predictions, step_test_data).item()}')

# Store the model and symbol_data for later use
    










# Now, symbol_data contains both dates and 'high' tensors for each symbol, with all series having the same length

# Now, symbol_data contains both the dates and 'high' tensors for each symbol
# For example, to access the data for a symbol named 'AAPL', you would do:
# aapl_data = symbol_data['AAPL']
# aapl_dates = aapl_data['dates']
# aapl_high_tensor = aapl_data['high_tensor']

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

