import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from AE import *
import matplotlib.pyplot as plt
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

parser = argparse.ArgumentParser(description='Train an autoencoder on synthetic data')
parser.add_argument('-hs', '--hidden_size', type=int, default=45, help='The hidden size of the LSTM')
parser.add_argument('-layers','--num_layers', type=int, default=1, help='The number of layers in the LSTM')
parser.add_argument('-epo','--epochs', type=int, default=10, help='The number of epochs to train the model')
parser.add_argument('-opt','--optimizer', type=str, default='Adam', help='The optimizer to use')
parser.add_argument('-lr','--learning_rate', type=float, default=0.005, help='The learning rate for the optimizer')
parser.add_argument('-gc','--grad_clip', type=int, default=0.5, help='The gradient clipping value')
parser.add_argument('-bs','--batch_size', type=int, default=32, help='The batch size for training')
parser.add_argument('-r','--random', action='store_true' , help='Whether to use a random seed')



args = parser.parse_args()

#Best Hyperparameters: {'learning_rate': 0.005, 'hidden_size': 45, 'num_layers': 1, 'batch_size': 32}
input_size = output_size = 1
hidden_size = args.hidden_size
num_layers = args.num_layers
epochs = args.epochs
optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'adadelta': torch.optim.Adadelta}
optimizer = optimizer_dict[args.optimizer]
learning_rate = args.learning_rate
grad_clip = args.grad_clip
batch_size = args.batch_size


if not args.random:    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

synthetic_data = torch.rand(10_000, 50, 1).to(device)

for j, x_ in enumerate(synthetic_data):
    i = np.random.randint(20, 30)
    mask = torch.tensor([1]*(i-5) + [0.1]*10 + [1]*(45-i)).to(device)
    synthetic_data[j] = (x_.flatten() * mask).reshape(50, 1)
    

total_size = len(synthetic_data)
train_size = int(0.6 * total_size)
validation_size = test_size = int(0.2 * total_size)

PLOT_EXAMPLE = True
if PLOT_EXAMPLE:
    random_index = np.random.randint(0, len(synthetic_data), (1,))
    random_index2 = np.random.randint(0, len(synthetic_data), (1,))
    data_sample = synthetic_data[random_index]
    data_sample2 = synthetic_data[random_index2]

    fig, axs = plt.subplots(2, figsize=(10, 6))
    axs[0].plot(data_sample.detach().cpu().numpy().flatten())
    axs[0].set_xlabel('Time') 
    axs[0].set_ylabel('Value') 
    axs[1].plot(data_sample2.detach().cpu().numpy().flatten())
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')
    fig.suptitle('Synthetic Data Example', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('synthetic_data_example.png')
    plt.show()


train_data, validation_data, test_data = random_split(synthetic_data, [train_size, validation_size, test_size])

# hyperparameters = {
#     'learning_rate': [0.001, 0.005, 0.01],
#     'hidden_size': [30, 40, 45],
#     'num_layers': [1, 2, 3],
#     'batch_size': [32, 64, 128],
#     # You can add more hyperparameters here
# }

# def grid_search(hyperparameters, train_data, validation_data):
#     best_val_loss = float('inf')
#     best_hyperparams = {}
#     # Generate all combinations of hyperparameters
#     for combination in itertools.product(*hyperparameters.values()):
#         hyperparams = dict(zip(hyperparameters.keys(), combination))
        
#         # Set up the model, dataloaders, and optimizer for this set of hyperparameters
#         batch_size = hyperparams['batch_size']
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#         validation_loader = DataLoader(validation_data, batch_size=batch_size)
        
#         model = AE(input_size=1, 
#                    hidden_size=hyperparams['hidden_size'], 
#                    num_layers=hyperparams['num_layers'], 
#                    output_size=1, 
#                    epochs=100, # You might want to adjust this for grid search
#                    optimizer=torch.optim.Adam, 
#                    learning_rate=hyperparams['learning_rate'], 
#                    grad_clip=0.5, 
#                    batch_size=batch_size).to(device)
        
#         # Train the model and evaluate it on the validation set
#         model.learn(train_loader, validation_loader)
#         val_loss = model.evaluate(validation_loader)  # You need to implement the evaluation method
        
#         # Check if this is the best performance we've seen so far
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_hyperparams = hyperparams

#     return best_hyperparams, best_val_loss



# best_hyperparams, best_val_loss = grid_search(hyperparameters, train_data, validation_data)
# print("Best Hyperparameters:", best_hyperparams)
# print("Best Validation Loss:", best_val_loss)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = AE(input_size, hidden_size, num_layers, output_size, epochs, optimizer, learning_rate, grad_clip, batch_size).to(device)

# Assuming your model's train method correctly handles device transfer inside
model.train()
model.learn(train_loader)
plt.plot(model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.show()

test_data_tensor = torch.stack([item for item in test_data.dataset]).to(device)
predictions = model(test_data_tensor)

difference = predictions - test_data_tensor




random_index = np.random.randint(0, len(test_data), (1,))
data_sample = test_data_tensor[random_index]
prediction_sample = predictions[random_index]

fig, axs = plt.subplots(2, figsize=(10, 6))
axs[0].plot(data_sample.detach().cpu().numpy().flatten())
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[1].plot(prediction_sample.detach().cpu().numpy().flatten())
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
fig.suptitle('Sample of Predictions and Original Data', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('predictions_vs_original.png')
plt.show()