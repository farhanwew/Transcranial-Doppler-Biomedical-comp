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

    # Prepare subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    categories = [
        ('High Quality (GOOD)', good_indices),
        ('Low Quality (BAD)', bad_indices),
        ('Mediocre (BORDERLINE)', borderline_indices),
        ('Corrupted (BAD)', bad_indices) # Using BAD again for 4th slot, or pick another random BAD
    ]

    for i, (title, indices) in enumerate(categories):
        ax = axes[i]
        if len(indices) > 0:
            # Pick a random index
            # For the 4th slot (Corrupted), try to pick a different one if possible, or just random
            idx = random.choice(indices)
            segment = all_segments[idx]
            
            ax.plot(segment)
            ax.set_title(title)
            ax.set_xlabel("Sample Index") # Using index since fs might vary per seg in this simple script
            ax.set_ylabel("CBFV [cm/s]")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No samples found", ha='center', va='center')
            ax.set_title(title)
            
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Quality samples visualization saved to {output_image}")

if __name__ == "__main__":
    visualize_quality_samples()
