import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

# Import VAE model definition
from vae_model import TCD_VAE

def visualize_vae_performance(
    dataset_path='tcd_dataset.npz', 
    labels_path='sqi_labels.npz', 
    model_path='vae_model.pth',
    output_image='vae_performance_analysis.png'
):
    
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path) or not os.path.exists(model_path):
        print("Error: Required files (dataset, labels, or model) not found.")
        return

    # 1. Load Data & Labels
    print("Loading data...")
    data = np.load(dataset_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    # Combine healthy & ICU segments
    all_segments = np.concatenate((data['healthy_valid'], data['icu_valid']))
    
    # Combine healthy & ICU SQI arrays
    # Note: valid_sqi in dataset are arrays of 1024. We need scalar score for display.
    all_sqi_arrays = np.concatenate((data['healthy_sqi'], data['icu_sqi']))
    
    # Calculate scalar SQI scores (median) for display
    all_sqi_scores = np.array([np.nanmedian(arr) if len(arr) > 0 else 0 for arr in all_sqi_arrays])

    # Labels
    # Use the first available scenario or default
    available_keys = labels.files
    label_suffix = ''
    for key in available_keys:
        if key.startswith('healthy_quality_labels_'):
            label_suffix = key.replace('healthy_quality_labels', '')
            break
    
    h_key = f'healthy_quality_labels{label_suffix}' if f'healthy_quality_labels{label_suffix}' in labels else 'healthy_quality_labels'
    i_key = f'icu_quality_labels{label_suffix}' if f'icu_quality_labels{label_suffix}' in labels else 'icu_quality_labels'
    
    all_labels = np.concatenate((labels[h_key], labels[i_key]))
    
    # Preprocessing function (same as in train_vae.py)
    def preprocess_segment(seg):
        min_val = np.min(seg)
        max_val = np.max(seg)
        if max_val > min_val:
            return (seg - min_val) / (max_val - min_val)
        return np.zeros_like(seg)

    # 2. Load VAE Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE model on {device}...")
    vae = TCD_VAE(input_length=1024, latent_dim=32).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # 3. Select Samples
    categories = {
        'GOOD': np.where(all_labels == 1)[0],
        'BORDERLINE': np.where(all_labels == -1)[0],
        'BAD': np.where(all_labels == 0)[0]
    }
    
    # Prepare plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (cat_name, indices) in enumerate(categories.items()):
        ax = axes[i]
        
        if len(indices) > 0:
            # Pick random sample
            idx = random.choice(indices)
            original_raw = all_segments[idx]
            sqi_score = all_sqi_scores[idx]
            
            # Normalize and prepare for VAE
            original_norm = preprocess_segment(original_raw)
            input_tensor = torch.Tensor(original_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            # Run VAE
            with torch.no_grad():
                recon_tensor, _, _ = vae(input_tensor)
            
            recon_signal = recon_tensor.squeeze().cpu().numpy()
            
            # Calculate MSE for this sample
            mse = np.mean((original_norm - recon_signal) ** 2)
            
            # Plot
            ax.plot(original_norm, label='Original (Norm)', alpha=0.7, color='blue')
            ax.plot(recon_signal, label='VAE Recon', alpha=0.7, color='red', linestyle='--')
            ax.set_title(f"Category: {cat_name} | MSE Error: {mse:.5f} | SQI: {sqi_score:.2f}")
            ax.legend(loc='upper right')
            ax.set_ylabel("Amplitude")
            if i == 2: ax.set_xlabel("Time [Samples]")
            ax.grid(True, alpha=0.3)
            
        else:
            ax.text(0.5, 0.5, f"No samples found for {cat_name}", ha='center', va='center')
            ax.set_title(f"Category: {cat_name}")

    plt.suptitle("VAE Reconstruction Performance by Quality Category", fontsize=16)
    plt.savefig(output_image)
    print(f"Visualization saved to {output_image}")

if __name__ == "__main__":
    visualize_vae_performance()
