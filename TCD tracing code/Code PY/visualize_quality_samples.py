import numpy as np
import matplotlib.pyplot as plt
import os
import random

def visualize_quality_samples(dataset_path='tcd_dataset.npz', labels_path='sqi_labels.npz', output_image='quality_samples.png'):
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path):
        print("Error: Data files missing.")
        return

    # Load data
    data = np.load(dataset_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    # Combine healthy and icu segments for sampling
    all_segments = np.concatenate((data['healthy_valid'], data['icu_valid']))
    all_labels = np.concatenate((labels['healthy_quality_labels'], labels['icu_quality_labels']))

    # Categorize segments
    good_indices = np.where(all_labels == 1)[0]
    bad_indices = np.where(all_labels == 0)[0]
    borderline_indices = np.where(all_labels == -1)[0]

    print(f"Found {len(good_indices)} GOOD segments.")
    print(f"Found {len(bad_indices)} BAD segments.")
    print(f"Found {len(borderline_indices)} BORDERLINE segments.")

    # Prepare subplots: 3 Rows (Good, Borderline, Bad), 3 Columns (Samples)
    num_samples_per_cat = 3
    fig, axes = plt.subplots(3, num_samples_per_cat, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    categories_config = [
        ('GOOD (High Quality)', good_indices),
        ('BORDERLINE (Mediocre)', borderline_indices),
        ('BAD (Low Quality)', bad_indices)
    ]

    for row_idx, (cat_name, indices) in enumerate(categories_config):
        # Select samples
        if len(indices) >= num_samples_per_cat:
            selected_indices = random.sample(list(indices), num_samples_per_cat)
        elif len(indices) > 0:
            # If fewer samples than needed, repeat with replacement
            selected_indices = [random.choice(indices) for _ in range(num_samples_per_cat)]
        else:
            selected_indices = []

        for col_idx in range(num_samples_per_cat):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(selected_indices):
                idx = selected_indices[col_idx]
                segment = all_segments[idx]
                
                ax.plot(segment)
                if col_idx == 1: # Title only on middle column for cleaner look
                    ax.set_title(f"{cat_name}", fontsize=12, fontweight='bold')
                
                # Add index as small text
                ax.text(0.02, 0.95, f"Idx: {idx}", transform=ax.transAxes, fontsize=8)
                
                if row_idx == 2: # X label only on bottom
                    ax.set_xlabel("Sample Index")
                if col_idx == 0: # Y label only on left
                    ax.set_ylabel("CBFV [cm/s]")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No sample", ha='center', va='center')
                if col_idx == 1: ax.set_title(f"{cat_name}")
            
    plt.suptitle("Random Samples by Quality Category", fontsize=16)
    plt.savefig(output_image)
    print(f"Quality samples visualization saved to {output_image}")

if __name__ == "__main__":
    visualize_quality_samples()
