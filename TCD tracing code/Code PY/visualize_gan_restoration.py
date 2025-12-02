import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

# Import models
from cyclegan_model import Generator

def visualize_gan_restoration(
    dataset_path='tcd_dataset.npz', 
    labels_path='sqi_labels.npz', 
    model_path='generator_AB.pth',
    output_image='gan_restoration_analysis.png'
):
    
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path) or not os.path.exists(model_path):
        print("Error: Required files (dataset, labels, or model) not found.")
        return

    # 1. Load Data & Labels
    print("Loading data...")
    data = np.load(dataset_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    # Extract specific segments
    h_segs = data['healthy_valid']
    i_segs = data['icu_valid']
    
    # Determine label keys (use first scenario found or default)
    available_keys = labels.files
    label_suffix = ''
    for key in available_keys:
        if key.startswith('healthy_quality_labels_'):
            label_suffix = key.replace('healthy_quality_labels', '') # keep underscore
            break
            
    h_labels = labels[f'healthy_quality_labels{label_suffix}'] if f'healthy_quality_labels{label_suffix}' in labels else labels['healthy_quality_labels']
    i_labels = labels[f'icu_quality_labels{label_suffix}'] if f'icu_quality_labels{label_suffix}' in labels else labels['icu_quality_labels']

    # Filter Borderline segments (-1)
    h_borderline_indices = np.where(h_labels == -1)[0]
    i_borderline_indices = np.where(i_labels == -1)[0]
    
    print(f"Found {len(h_borderline_indices)} Healthy Borderline segments.")
    print(f"Found {len(i_borderline_indices)} ICU Borderline segments.")

    # 2. Load GAN Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Generator on {device}...")
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    # Helper to restore
    def restore_segment(seg):
        # Normalize per segment to [0, 1] then to [-1, 1] for GAN
        min_val = np.min(seg)
        max_val = np.max(seg)
        if max_val > min_val:
            seg_norm = (seg - min_val) / (max_val - min_val)
            seg_gan = seg_norm * 2 - 1
        else:
            return np.zeros_like(seg)
            
        tensor_in = torch.Tensor(seg_gan).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            tensor_out = G(tensor_in)
        
        # Map back to [0, 1]
        seg_out = (tensor_out.squeeze().cpu().numpy() + 1) / 2
        return seg_out, seg_norm # Return both normalized versions for comparison

    # 3. Select and Plot Samples
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    
    # Columns: Left=Healthy, Right=ICU
    
    # --- Healthy Samples ---
    if len(h_borderline_indices) > 0:
        indices = random.sample(list(h_borderline_indices), min(num_samples, len(h_borderline_indices)))
        for i, idx in enumerate(indices):
            original, _ = restore_segment(h_segs[idx]) # Dummy call just to normalize? No, need to run restoration
            restored, original_norm = restore_segment(h_segs[idx])
            
            ax = axes[i, 0]
            ax.plot(original_norm, label='Original (Borderline)', alpha=0.6, color='blue')
            ax.plot(restored, label='GAN Restored', alpha=0.8, color='green', linestyle='--')
            ax.set_title(f"Healthy Borderline (Idx: {idx})")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            if i == num_samples-1: ax.set_xlabel("Sample Index")
    else:
        for i in range(num_samples):
            axes[i, 0].text(0.5, 0.5, "No Healthy Borderline Samples", ha='center')

    # --- ICU Samples ---
    if len(i_borderline_indices) > 0:
        indices = random.sample(list(i_borderline_indices), min(num_samples, len(i_borderline_indices)))
        for i, idx in enumerate(indices):
            restored, original_norm = restore_segment(i_segs[idx])
            
            ax = axes[i, 1]
            ax.plot(original_norm, label='Original (Borderline)', alpha=0.6, color='red')
            ax.plot(restored, label='GAN Restored', alpha=0.8, color='green', linestyle='--')
            ax.set_title(f"ICU Borderline (Idx: {idx})")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            if i == num_samples-1: ax.set_xlabel("Sample Index")
    else:
        for i in range(num_samples):
            axes[i, 1].text(0.5, 0.5, "No ICU Borderline Samples", ha='center')

    plt.suptitle("GAN Restoration: Borderline to Good-like (Healthy vs ICU)", fontsize=16)
    plt.savefig(output_image)
    print(f"Visualization saved to {output_image}")

if __name__ == "__main__":
    visualize_gan_restoration()
