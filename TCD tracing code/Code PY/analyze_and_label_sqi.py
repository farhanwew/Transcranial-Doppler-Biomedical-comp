import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_label_sqi(dataset_filepath='tcd_dataset.npz'):
    if not os.path.exists(dataset_filepath):
        print(f"Error: Dataset file not found at {dataset_filepath}")
        return

    # Load data
    data = np.load(dataset_filepath, allow_pickle=True)
    
    # Extract SQI arrays for valid segments
    healthy_sqi_arrays = np.array(data['healthy_sqi'])
    icu_sqi_arrays = np.array(data['icu_sqi'])

    # Calculate median SQI for each segment
    # This is SQI_seg (one scalar value per segment) as per step.md C.1
    # Handle cases where SQI array might be empty or contain NaNs for median calculation
    healthy_sqi_seg = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in healthy_sqi_arrays])
    icu_sqi_seg = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in icu_sqi_arrays])

    # Filter out NaN SQI_seg values that might result from empty SQI arrays
    healthy_sqi_seg = healthy_sqi_seg[~np.isnan(healthy_sqi_seg)]
    icu_sqi_seg = icu_sqi_seg[~np.isnan(icu_sqi_seg)]

    all_sqi_seg = np.concatenate((healthy_sqi_seg, icu_sqi_seg))
    
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
    # plt.show() # In Colab, plt.show() is often not needed if it's the last line, but explicitly saving is better
    plt.savefig('sqi_seg_distribution.png')
    print("SQI_seg distribution histogram saved to sqi_seg_distribution.png")


    # --- Determining Thresholds and Creating Labels ---
    # These thresholds are examples and should be adjusted based on the histogram.
    # From step.md: T_high (e.g., 80%), T_low (e.g., 40%)
    
    # Placeholder thresholds based on common practice or initial visual inspection
    T_high = 80 # Example: Segments with median SQI >= 80 are GOOD
    T_low = 40  # Example: Segments with median SQI <= 40 are BAD

    # --- Create quality_labels_sqi for Healthy segments ---
    # Re-calculate sqi_seg based on original arrays for consistent indexing
    _healthy_sqi_seg_for_labeling = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in data['healthy_sqi']])
    healthy_quality_labels = np.full(len(_healthy_sqi_seg_for_labeling), -1, dtype=int) # -1 for BORDERLINE initially
    healthy_quality_labels[_healthy_sqi_seg_for_labeling >= T_high] = 1 # GOOD
    healthy_quality_labels[_healthy_sqi_seg_for_labeling <= T_low] = 0 # BAD

    # --- Create quality_labels_sqi for ICU segments ---
    _icu_sqi_seg_for_labeling = np.array([np.nanmedian(arr) if arr.size > 0 else np.nan for arr in data['icu_sqi']])
    icu_quality_labels = np.full(len(_icu_sqi_seg_for_labeling), -1, dtype=int) # -1 for BORDERLINE initially
    icu_quality_labels[_icu_sqi_seg_for_labeling >= T_high] = 1 # GOOD
    icu_quality_labels[_icu_sqi_seg_for_labeling <= T_low] = 0 # BAD

    print(f"\nUsing T_high={T_high} and T_low={T_low} for labeling:")
    print(f"  Healthy: GOOD={np.sum(healthy_quality_labels == 1)}, BAD={np.sum(healthy_quality_labels == 0)}, BORDERLINE={np.sum(healthy_quality_labels == -1)}")
    print(f"  ICU: GOOD={np.sum(icu_quality_labels == 1)}, BAD={np.sum(icu_quality_labels == 0)}, BORDERLINE={np.sum(icu_quality_labels == -1)}")

    # Save quality labels back to the dataset
    # It's better to save a new file for labels and thresholds to avoid modifying the original data npz
    sqi_labels_filepath = 'sqi_labels.npz'
    np.savez(sqi_labels_filepath, 
             healthy_sqi_seg=healthy_sqi_seg, healthy_quality_labels=healthy_quality_labels,
             icu_sqi_seg=icu_sqi_seg, icu_quality_labels=icu_quality_labels,
             T_high=T_high, T_low=T_low)
    print(f"SQI_seg values and quality labels saved to {sqi_labels_filepath}")

if __name__ == "__main__":
    analyze_and_label_sqi()