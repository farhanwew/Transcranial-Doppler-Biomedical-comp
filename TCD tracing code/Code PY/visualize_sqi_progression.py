import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sqi_progression(dataset_path='tcd_dataset.npz', output_image='sqi_progression.png'):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    
    # 1. Combine all segments and their SQI arrays
    # Note: 'valid_sqi' in the npz contains the full arrays (1024 samples)
    all_cbfv = np.concatenate((data['healthy_valid'], data['icu_valid']))
    all_sqi_arrays = np.concatenate((data['healthy_sqi'], data['icu_sqi']))
    
    # 2. Calculate scalar SQI score for sorting (Median of the array)
    # Using nanmedian to handle potential NaNs in the SQI trace
    print("Calculating SQI scores for sorting...")
    sqi_scores = np.array([np.nanmedian(arr) if len(arr) > 0 else 0 for arr in all_sqi_arrays])
    
    # 3. Find samples closest to target SQI values
    target_sqis = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [f"Target SQI: {val}" for val in target_sqis]
    
    sample_indices = []
    
    print("Finding closest samples to targets...")
    for target in target_sqis:
        # Find index of the segment with SQI closest to target
        # abs(sqi_scores - target) gives distance, argmin gives index of min distance
        idx = (np.abs(sqi_scores - target)).argmin()
        sample_indices.append(idx)
        actual_score = sqi_scores[idx]
        print(f"  Target {target}: Found sample with SQI {actual_score:.4f}")

    # 5. Plotting
    print("Generating visualization...")
    fig, axes = plt.subplots(5, 2, figsize=(15, 15), sharex='col')
    plt.subplots_adjust(hspace=0.4)
    
    for i, idx in enumerate(sample_indices):
        cbfv_segment = all_cbfv[idx]
        sqi_segment = all_sqi_arrays[idx]
        score = sqi_scores[idx]
        
        # Calculate Time Axis (assuming fs ~ 217 Hz for visualization)
        fs = 217.0 
        t_axis = np.arange(len(cbfv_segment)) / fs

        # Plot CBFV Envelope (Left Column)
        ax_env = axes[i, 0]
        ax_env.plot(t_axis, cbfv_segment, color='blue', linewidth=1)
        ax_env.set_title(f"{labels[i]} (Actual: {score:.2f}) - CBFV Envelope")
        ax_env.set_ylabel("Velocity [cm/s]")
        ax_env.grid(True, alpha=0.3)
        
        # Plot SQI Trace (Right Column)
        ax_sqi = axes[i, 1]
        ax_sqi.plot(t_axis, sqi_segment * 100, color='red', linewidth=1.5)
        ax_sqi.set_title(f"SQI Trace")
        ax_sqi.set_ylabel("SQI [%]")
        ax_sqi.set_ylim([-5, 105])
        ax_sqi.grid(True, alpha=0.3)
        
        # Set X label only for bottom rows
        if i == 4:
            ax_env.set_xlabel("Time [s]")
            ax_sqi.set_xlabel("Time [s]")

    plt.suptitle("Visual Progression of Signal Quality (Targeted SQI Levels)", fontsize=16)
    plt.savefig(output_image)
    print(f"Visualization saved to {output_image}")

if __name__ == "__main__":
    visualize_sqi_progression()
