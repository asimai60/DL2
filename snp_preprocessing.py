import pandas as pd
import torch

def load_data():
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')

    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date'])

    df = df[['symbol', 'date', 'high']]
    return df

def prepare_data(device, sequence_length=30):
    df = load_data()

    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
    sample_size = sequence_length  # e.g., 30 for monthly data
    # Initialize a dictionary to store processed data for each symbol

    num_subsequences = full_date_range.size // sample_size

    long_sequences = []
    all_subsequences = []  # List to store all subsequences
    all_real_next_values_per_subsequence = []  # List to store all predictions per subsequence
    min_max_dict = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        # Reindex the DataFrame to ensure it covers the full date range
        symbol_df.set_index('date', inplace=True)
        symbol_df = symbol_df.reindex(full_date_range)
        
        # Rename the index to 'date' after reindexing
        symbol_df.index.name = 'date'
        
        # Use bfill() or ffill() to fill missing values after reindexing
        symbol_df = symbol_df.bfill()#.ffill()  # First backward fill, then forward fill to cover all gaps
        
        
        # Normalize the 'high' column using Min-Max scaling
        min_val = symbol_df['high'].min()
        max_val = symbol_df['high'].max()
        min_max_dict[symbol] = (min_val, max_val)
        symbol_df['high'] = (symbol_df['high'] - min_val) / (max_val - min_val)
        
        full_sequence_high_tensor = torch.tensor(symbol_df['high'].values, dtype=torch.float)
        sequence_length = full_sequence_high_tensor.size(0)
        subsequence_length = sequence_length // num_subsequences
        trimmed_length = subsequence_length * num_subsequences
        full_sequence_high_tensor = full_sequence_high_tensor[:trimmed_length]

        subsequences = full_sequence_high_tensor.split(subsequence_length)
        
        subsequences = [((subsequence - torch.min(subsequence))/(torch.max(subsequence)-torch.min(subsequence))) for subsequence in subsequences]

        real_next_values_per_subsequence = [subsequence[-1] for subsequence in subsequences]

        
        subsequences = [subsequence[:-1] for subsequence in subsequences]

        
        
        all_real_next_values_per_subsequence.extend(real_next_values_per_subsequence)
        
        all_subsequences.extend(subsequences)
        long_sequences.append(full_sequence_high_tensor)
        
    long_sequences = torch.stack(long_sequences).unsqueeze(-1).to(device)
    data = torch.stack(all_subsequences).unsqueeze(-1).to(device)
    dic_keys = [(symbol,i) for symbol in df['symbol'].unique() for i in range(num_subsequences)]

    real_next_values = torch.stack(all_real_next_values_per_subsequence).to(device)
    data_dict = {key : (data[i], real_next_values[i],min_max_dict[key[0]]) for i, key in enumerate(dic_keys)}

    
    split_ratio = 0.8
    train_data = data[:int(split_ratio * data.size(0))]
    train_real_next_values = real_next_values[:int(split_ratio * real_next_values.size(0))]
    
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    train_real_next_values = train_real_next_values[indices]
    
    test_data = data[int(split_ratio * data.size(0)):]
    test_real_next_values = real_next_values[int(split_ratio * real_next_values.size(0)):]

    return train_data, test_data, train_real_next_values, test_real_next_values, data_dict, long_sequences