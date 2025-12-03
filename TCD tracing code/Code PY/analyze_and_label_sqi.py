import argparse # Added import
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_label_sqi(dataset_filepath='tcd_dataset.npz', t_high=0.85, t_low=0.70):
    if not os.path.exists(dataset_filepath):
        print(f"Error: Dataset file not found at {dataset_filepath}")
        return

    # Load data
    data = np.load(dataset_filepath, allow_pickle=True)
    
    # ... (Rest of the data loading and histogram plotting code remains the same) ...
    # Extract SQI arrays for valid segments
    healthy_sqi_arrays = np.array(data['healthy_sqi'])
    icu_sqi_arrays = np.array(data['icu_sqi'])

    # Calculate median SQI for each segment
    healthy_sqi_seg = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in healthy_sqi_arrays])
    icu_sqi_seg = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in icu_sqi_arrays])

    # Filter out NaN SQI_seg values that might result from empty SQI arrays for stats
    healthy_sqi_seg_clean = healthy_sqi_seg[~np.isnan(healthy_sqi_seg)]
    icu_sqi_seg_clean = icu_sqi_seg[~np.isnan(icu_sqi_seg)]

    all_sqi_seg = np.concatenate((healthy_sqi_seg_clean, icu_sqi_seg_clean))
    
    if len(all_sqi_seg) == 0:
        print("No valid SQI_seg data found to analyze after median calculation.")
        return

    print(f"Total valid segments for SQI_seg analysis: {len(all_sqi_seg)}")
    print(f"SQI_seg distribution (all segments):")
    print(f"  Min: {np.min(all_sqi_seg):.2f}")
    print(f"  Max: {np.max(all_sqi_seg):.2f}")
    print(f"  Mean: {np.mean(all_sqi_seg):.2f}")
    print(f"  Median: {np.median(all_sqi_seg):.2f}")
    print(f"  Std Dev: {np.std(all_sqi_seg):.2f}")
    print(f"  25th Percentile: {np.percentile(all_sqi_seg, 25):.2f}")
    print(f"  75th Percentile: {np.percentile(all_sqi_seg, 75):.2f}")

    # --- Plotting Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_sqi_seg, bins=50, alpha=0.7, label='All Valid Segments SQI_seg')
    plt.title('Distribution of SQI_seg per Segment')
    plt.xlabel('SQI_seg Value (Median of Beat-wise SQI)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig('sqi_seg_distribution.png')
    print("SQI_seg distribution histogram saved to sqi_seg_distribution.png")


    # --- Determining Thresholds and Creating Labels ---
    
    # Use provided arguments
    T_high = t_high
    T_low = t_low
    
    # Define threshold scenarios (Just one 'Custom' scenario based on args now, or keep list if needed)
    # To keep it simple and consistent with the request, we'll just use the args as the single scenario 'Custom'
    # But to maintain compatibility with train_classifier which expects keys, we can save it as 'Custom' or 'Default'.
    
    threshold_scenarios = [
        {'name': 'Custom', 'T_high': T_high, 'T_low': T_low} 
    ]
    
    labels_to_save = {
        'healthy_sqi_seg': healthy_sqi_seg,
        'icu_sqi_seg': icu_sqi_seg,
        'scenarios': threshold_scenarios 
    }

    for scenario in threshold_scenarios:
        name = scenario['name']
        T_high = scenario['T_high']
        T_low = scenario['T_low']
        
        print(f"\n--- Processing Scenario: {name} (High={T_high}, Low={T_low}) ---")

        # --- Create quality_labels for Healthy segments ---
        # Re-calculate sqi_seg based on original arrays for consistent indexing
        # Note: healthy_sqi_seg computed above already aligns with indices, just includes NaNs which is correct for indexing
        _healthy_sqi_seg_for_labeling = healthy_sqi_seg
        
        healthy_labels = np.full(len(_healthy_sqi_seg_for_labeling), -1, dtype=int) # Default to BORDERLINE
        healthy_labels[_healthy_sqi_seg_for_labeling >= T_high] = 1 # GOOD
        healthy_labels[_healthy_sqi_seg_for_labeling <= T_low] = 0 # BAD
        # Explicitly label NaN SQI segments as BAD
        healthy_labels[np.isnan(_healthy_sqi_seg_for_labeling)] = 0 # BAD
        
        # --- Create quality_labels for ICU segments ---
        _icu_sqi_seg_for_labeling = icu_sqi_seg
        
        icu_labels = np.full(len(_icu_sqi_seg_for_labeling), -1, dtype=int) # Default to BORDERLINE
        icu_labels[_icu_sqi_seg_for_labeling >= T_high] = 1 # GOOD
        icu_labels[_icu_sqi_seg_for_labeling <= T_low] = 0 # BAD
        # Explicitly label NaN SQI segments as BAD
        icu_labels[np.isnan(_icu_sqi_seg_for_labeling)] = 0 # BAD

        # Statistics helper
        def print_stats(label_name, labels_arr, sqi_arr):
            print(f"  {label_name}:")
            for cat_val, cat_name in zip([1, 0, -1], ['GOOD', 'BAD', 'BORDERLINE']):
                indices = np.where(labels_arr == cat_val)[0]
                count = len(indices)
                if count > 0:
                    sqi_vals = sqi_arr[indices]
                    mean_val = np.nanmean(sqi_vals)
                    median_val = np.nanmedian(sqi_vals)
                    print(f"    {cat_name}: Count={count}, Mean SQI={mean_val:.4f}, Median SQI={median_val:.4f}")
                else:
                    print(f"    {cat_name}: Count=0")

        print_stats("Healthy", healthy_labels, _healthy_sqi_seg_for_labeling)
        print_stats("ICU", icu_labels, _icu_sqi_seg_for_labeling)
        
        # Add to save dict with specific keys
        # Saving as 'healthy_quality_labels_Custom'
        labels_to_save[f'healthy_quality_labels_{name}'] = healthy_labels
        labels_to_save[f'icu_quality_labels_{name}'] = icu_labels

    # Save quality labels back to the dataset
    sqi_labels_filepath = 'sqi_labels.npz'
    np.savez(sqi_labels_filepath, **labels_to_save)
    print(f"\nSQI values and quality labels (Scenario: Custom) saved to {sqi_labels_filepath}")

    # --- DEBUG: Inspect Sample SQI Arrays ---
    print("\n--- DEBUG: Sample SQI Arrays (from Scenario: Custom) ---")
    h_labels_debug = labels_to_save['healthy_quality_labels_Custom']
    
    categories = {'GOOD': 1, 'BORDERLINE': -1, 'BAD': 0}
    
    # ... (Debug plotting code remains the same, just using h_labels_debug) ...
    print("\nGenerating debug plots for sample segments...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (cat_name, cat_val) in enumerate(categories.items()):
        indices = np.where(h_labels_debug == cat_val)[0]
        
        ax_cbfv = axes[i, 0]
        ax_sqi = axes[i, 1]
        
        if len(indices) > 0:
            # Pick random
            idx = np.random.choice(indices)
            # Need to load original segments for plotting, data is loaded at top
            # data['healthy_valid'] contains segments
            cbfv_seg = data['healthy_valid'][idx]
            sqi_arr = data['healthy_sqi'][idx]
            
            median_val = np.nanmedian(sqi_arr) if len(sqi_arr) > 0 else 0
            mean_val = np.nanmean(sqi_arr) if len(sqi_arr) > 0 else 0
            
            print(f"\nCategory: {cat_name} (Index {idx})")
            print(f"  Median Score: {median_val:.4f}")
            print(f"  Mean Score:   {mean_val:.4f}")
            
            # CBFV Plot
            ax_cbfv.plot(cbfv_seg, 'b')
            ax_cbfv.set_title(f"{cat_name} - CBFV Envelope")
            ax_cbfv.set_ylabel("Velocity [cm/s]")
            ax_cbfv.grid(True, alpha=0.3)
            
            # SQI Plot
            ax_sqi.plot(sqi_arr * 100, 'r')
            ax_sqi.set_title(f"{cat_name} - SQI Trace")
            ax_sqi.set_ylabel("SQI [%]")
            ax_sqi.set_ylim([-5, 105])
            ax_sqi.grid(True, alpha=0.3)
        else:
            ax_cbfv.text(0.5, 0.5, "No Data", ha='center')
            ax_sqi.text(0.5, 0.5, "No Data", ha='center')
            
    plt.suptitle("Debug: Sample Segments by Quality Label", fontsize=16)
    plt.savefig("debug_samples.png")
    print("Debug plot saved to debug_samples.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SQI and generate quality labels.")
    parser.add_argument('--dataset_filepath', type=str, default='tcd_dataset.npz', help='Path to the dataset file.')
    parser.add_argument('--t_high', type=float, default=0.85, help='Threshold for GOOD quality (>= t_high).')
    parser.add_argument('--t_low', type=float, default=0.70, help='Threshold for BAD quality (<= t_low).')
    args = parser.parse_args()
    
    analyze_and_label_sqi(dataset_filepath=args.dataset_filepath, t_high=args.t_high, t_low=args.t_low)