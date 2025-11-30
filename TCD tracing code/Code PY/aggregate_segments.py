import os
import numpy as np
import glob

def aggregate_segments(input_dir='processed_segments', output_file='tcd_dataset.npz'):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    print(f"Scanning {input_dir} for .npz files...")
    npz_files = glob.glob(os.path.join(input_dir, "*_segments.npz"))
    
    if not npz_files:
        print("No .npz files found. Did you run process_dataset.py?")
        return

    all_healthy_segments = []
    all_healthy_sqi = []
    
    all_icu_segments = []
    all_icu_sqi = []

    print(f"Found {len(npz_files)} files. Aggregating...")

    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            
            # Check class label (stored as a 0-d array or string in npz)
            # We saved it as class_label='Healthy' or 'ICU'
            label = str(data['class_label'])
            
            valid_cbfv = data['valid_cbfv']
            valid_sqi = data['valid_sqi'] # This is now FULL array (1024 samples)
            
            if len(valid_cbfv) > 0:
                if label == 'Healthy':
                    all_healthy_segments.extend(valid_cbfv)
                    all_healthy_sqi.extend(valid_sqi)
                elif label == 'ICU':
                    all_icu_segments.extend(valid_cbfv)
                    all_icu_sqi.extend(valid_sqi)
                else:
                    print(f"Warning: Unknown label '{label}' in {f}")
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Convert lists to arrays
    healthy_valid = np.array(all_healthy_segments)
    healthy_sqi = np.array(all_healthy_sqi)
    icu_valid = np.array(all_icu_segments)
    icu_sqi = np.array(all_icu_sqi)

    print("\n--- Aggregation Summary ---")
    print(f"Total Healthy Segments: {len(healthy_valid)}")
    print(f"Total ICU Segments: {len(icu_valid)}")

    if len(healthy_valid) == 0 and len(icu_valid) == 0:
        print("No valid segments found in any file.")
        return

    # Save to full dataset file
    # Using keys compatible with our training scripts
    np.savez(output_file, 
             healthy_valid=healthy_valid, 
             healthy_sqi=healthy_sqi,
             icu_valid=icu_valid, 
             icu_sqi=icu_sqi)
    
    print(f"Full dataset saved to: {output_file}")

if __name__ == "__main__":
    aggregate_segments()
