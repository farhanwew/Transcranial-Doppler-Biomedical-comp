import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

def split_dataset(dataset_path='tcd_dataset.npz', labels_path='sqi_labels.npz', test_size=0.2, random_state=42):
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path):
        print("Error: Input files not found.")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    sqi_data = np.load(labels_path, allow_pickle=True)

    # Extract segments and class labels
    h_segs = data['healthy_valid']
    i_segs = data['icu_valid']
    
    # Extract SQI labels (handling multiple scenarios if present, or default)
    # We need to split ALL scenario labels consistent with the segments
    
    # Create a unified list of indices to split, then apply to all arrays
    # Healthy
    h_indices = np.arange(len(h_segs))
    h_train_idx, h_test_idx = train_test_split(h_indices, test_size=test_size, random_state=random_state)
    
    # ICU
    i_indices = np.arange(len(i_segs))
    i_train_idx, i_test_idx = train_test_split(i_indices, test_size=test_size, random_state=random_state)
    
    def save_split(filename, h_idx, i_idx):
        split_data = {}
        
        # 1. Segments
        split_data['healthy_valid'] = h_segs[h_idx]
        split_data['icu_valid'] = i_segs[i_idx]
        
        # 2. Original SQI values (if needed)
        split_data['healthy_sqi'] = data['healthy_sqi'][h_idx]
        split_data['icu_sqi'] = data['icu_sqi'][i_idx]
        
        # 3. SQI Labels (for all scenarios found in labels file)
        for key in sqi_data.files:
            if key.startswith('healthy_quality_labels'):
                split_data[key] = sqi_data[key][h_idx]
            elif key.startswith('icu_quality_labels'):
                split_data[key] = sqi_data[key][i_idx]
            elif key in ['T_high', 'T_low', 'scenarios']: # Metadata
                split_data[key] = sqi_data[key]
                
        np.savez(filename, **split_data)
        print(f"Saved {filename}: {len(h_idx)} Healthy, {len(i_idx)} ICU segments.")

    save_split('tcd_train.npz', h_train_idx, i_train_idx)
    save_split('tcd_test.npz', h_test_idx, i_test_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split TCD dataset into Train and Test sets.")
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset to include in the test split.')
    args = parser.parse_args()
    
    split_dataset(test_size=args.test_size)
