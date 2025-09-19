import pandas as pd
import os
import numpy as np


def process_csv_files(root_dir='utils/splits_train_val_test'):
    # Walk through all directories in splits folder
    for dataset_dir in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_dir)
        
        if not os.path.isdir(dataset_path):
            continue
            
        for file in os.listdir(dataset_path):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(dataset_path, file)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get validation column values (non-NA)
            val_values = df['val'].dropna().values
            train_values = df['train'].dropna().values
            
            # Split validation values into two parts
            n_test = len(val_values) // 2
            test_values = val_values[:n_test]
            val_values = val_values[n_test:]
            
            # Find maximum length
            max_len = max(len(train_values), len(val_values), len(test_values))
            
            # Pad arrays with NaN to match max_len
            train_padded = np.pad(train_values, (0, max_len - len(train_values)), 
                                constant_values=np.nan, mode='constant')
            val_padded = np.pad(val_values, (0, max_len - len(val_values)), 
                              constant_values=np.nan, mode='constant')
            test_padded = np.pad(test_values, (0, max_len - len(test_values)), 
                               constant_values=np.nan, mode='constant')
            
            # Create new DataFrame with padded arrays
            new_df = pd.DataFrame({
                'train': train_padded,
                'val': val_padded,
                'test': test_padded
            })
            
            # Save modified DataFrame
            new_df.to_csv(file_path, index=False)
            print(f"Processed {file_path}")

if __name__ == "__main__":
    process_csv_files()