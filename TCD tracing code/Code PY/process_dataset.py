import os
import numpy as np
import glob
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import random
import argparse

# Import local modules
from preprocessing import preprocess_time_vector, segment_echo_signal, load_tcd_data, segment_cbfv_into_windows
from spectrogram_utils import generate_spectrogram
from tracing import spectrogram_tracing
from postprocessing import postprocess_cbfv

# Parameters
FC = 1.75e6
SS = 1540
ANGLE = 0
SIZE_SP_FILT = [3, 3]

def find_recordings(healthy_folder_path, icu_folder_path):
    """
    Finds TCD recordings based on the specified structure.
    Healthy: *.txt
    ICU: *_MCA.txt
    """
    healthy_files = []
    if os.path.exists(healthy_folder_path):
        all_txt = glob.glob(os.path.join(healthy_folder_path, "*.txt"))
        healthy_files = sorted(all_txt)
    
    icu_files = []
    if os.path.exists(icu_folder_path):
        icu_files = sorted(glob.glob(os.path.join(icu_folder_path, "*_MCA.txt")))
        
    return healthy_files, icu_files

def process_recording(filepath):
    """
    Runs the full extraction pipeline on a single file.
    Returns:
        valid_windows (list): List of clean CBFV segments (1024 samples)
        valid_sqi_arrays (list): List of full SQI arrays (1024 samples) corresponding to valid windows
        corrupted_windows (list): List of rejected CBFV segments
        corrupted_reasons (list): List of strings explaining rejection (e.g., 'NaN', 'Flat')
    """
    try:
        raw_data = load_tcd_data(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], [], [], []

    # Preprocess time
    t = preprocess_time_vector(raw_data['t'].values)
    
    # Segment echo data (Original large segments for processing)
    tcd_echo_data = segment_echo_signal(raw_data['I'].values, raw_data['Q'].values, t)
    
    full_cbfv_envelope = []
    full_sqi_envelope = [] # Capture SQI
    
    # Process each large segment to get the envelope
    for i in range(len(tcd_echo_data['t'])):
        iq_signal = tcd_echo_data['IQ'][i]
        fs_echo = tcd_echo_data['fs'][i]
        t_echo = tcd_echo_data['t'][i]
        
        # Spectrogram
        f_sp, _, sp_seg, fs_sp = generate_spectrogram(iq_signal, fs_echo, t_echo)
        
        # Velocity
        v_spectrogram = 100 * f_sp * SS / (2 * FC * np.cos(np.deg2rad(ANGLE)))
        
        # Image Proc
        sp_seg = np.log2(sp_seg + 1e-9)
        max_val = np.max(sp_seg)
        if max_val != 0:
            sp_seg = sp_seg / max_val
        sp_seg = signal.medfilt2d(sp_seg, kernel_size=SIZE_SP_FILT)
        
        # Otsu Indices
        ix_exclude_otsu = []
        if len(tcd_echo_data['ix_low_freq'][i]) > 0:
            ratio = fs_sp / fs_echo
            ix_exclude_otsu = np.round(tcd_echo_data['ix_low_freq'][i] * ratio).astype(int)
            
        # Tracing (Adaptive only for SQA dataset)
        # Now capturing cbfv_sqi_curr
        cbfv_raw, cbfv_sqi_curr, _, _ = spectrogram_tracing(v_spectrogram, sp_seg, fs_sp, ix_exclude_otsu)
        cbfv_smooth = postprocess_cbfv(cbfv_raw, fs_sp)
        
        full_cbfv_envelope.extend(cbfv_smooth)
        full_sqi_envelope.extend(cbfv_sqi_curr)
        
    # Segment into 1024-sample windows
    cbfv_windows = segment_cbfv_into_windows(np.array(full_cbfv_envelope), window_length=1024, overlap=0)
    sqi_windows = segment_cbfv_into_windows(np.array(full_sqi_envelope), window_length=1024, overlap=0)
    
    # Filter NaNs / Flat lines (Simple Hard Rule Filter)
    # BUT keep corrupted ones for inspection
    valid_cbfv = []
    valid_sqi_arrays = [] # Store FULL SQI array per segment
    
    corrupted_cbfv = []
    corrupted_reasons = []
    
    # Ensure lengths match
    min_len = min(len(cbfv_windows), len(sqi_windows))
    
    for i in range(min_len):
        w_cbfv = cbfv_windows[i]
        w_sqi = sqi_windows[i]
        
        # Check for corruption
        is_nan = np.isnan(w_cbfv).any()
        is_flat = np.std(w_cbfv) < 1e-3
        
        if is_nan:
            corrupted_cbfv.append(w_cbfv)
            corrupted_reasons.append('NaN')
        elif is_flat:
            corrupted_cbfv.append(w_cbfv)
            corrupted_reasons.append('Flat')
        else:
            valid_cbfv.append(w_cbfv)
            # Save the FULL SQI array for visualization (like Figure 5)
            valid_sqi_arrays.append(w_sqi)
        
    return valid_cbfv, valid_sqi_arrays, corrupted_cbfv, corrupted_reasons, fs_sp

def visualize_single_segment(segment, sqi_segment=None, title="CBFV Segment", segment_idx=None, reason=None, fs=None):
    """
    Visualizes a single 1024-sample CBFV segment and optional SQI plot.
    Matches the style of the reference figure (Envelope + SQI).
    """
    full_title = title
    if segment_idx is not None:
        full_title += f" - Idx {segment_idx}"
    if reason:
        full_title += f" ({reason})"
    
    # Time axis calculation
    if fs:
        t_axis = np.arange(len(segment)) / fs
        x_label = "Time [s]"
    else:
        t_axis = np.arange(len(segment))
        x_label = "Sample Index"
        
    if sqi_segment is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Plot Envelope
        ax1.plot(t_axis, segment, 'r', linewidth=1.5, label='Envelope')
        ax1.set_ylabel("CBFV [cm/s]")
        ax1.set_title(full_title)
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Plot SQI
        ax2.plot(t_axis, sqi_segment * 100, 'r', linewidth=1.5, label='SQI') # Scale to %
        ax2.set_ylabel("SQI [%]")
        ax2.set_ylim([-5, 105]) # Fixed range for SQI
        ax2.set_xlabel(x_label)
        ax2.set_title("Signal Quality Index")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, segment)
        plt.title(full_title)
        plt.xlabel(x_label)
        plt.ylabel("CBFV [cm/s]")
        plt.grid(True)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process TCD recordings to create a segmented dataset.")
    parser.add_argument('--healthy_path', type=str, required=True,
                        help='Path to the "Healthy Subjects" data folder.')
    parser.add_argument('--icu_path', type=str, required=True,
                        help='Path to the "ICU Patients" data folder.')
    args = parser.parse_args()
    
    healthy_data_path = args.healthy_path
    icu_data_path = args.icu_path
    
    print(f"Using Healthy data path: {os.path.abspath(healthy_data_path)}")
    print(f"Using ICU data path: {os.path.abspath(icu_data_path)}")

    print("Searching for data...")
    healthy_files, icu_files = find_recordings(healthy_data_path, icu_data_path)
            
    print(f"Found {len(healthy_files)} Healthy recordings.")
    print(f"Found {len(icu_files)} ICU recordings (filtered by _MCA).")
    
    # Create output directory for processed segments
    output_dir = "processed_segments"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processed segments will be saved to: {os.path.abspath(output_dir)}")
    
    # Store sample data and their fs for visualization later
    sample_healthy_valid = []
    sample_healthy_sqi = []
    sample_healthy_fs = None # To store fs for healthy samples
    
    sample_icu_valid = []
    sample_icu_sqi = []
    sample_icu_fs = None # To store fs for icu samples
    
    sample_corrupted = []
    sample_corrupted_reason = []
    sample_corrupted_fs = None # To store fs for corrupted samples
    
    # Process Healthy
    for f in healthy_files:
        filename = os.path.basename(f)
        print(f"Processing {filename} from {healthy_data_path}...")
        v_cbfv, v_sqi, c_cbfv, c_reasons, fs_rec = process_recording(f)
        
        if len(v_cbfv) > 0:
            # Keep some samples for viz
            if len(sample_healthy_valid) < 5:
                sample_healthy_valid.extend(v_cbfv[:2])
                sample_healthy_sqi.extend(v_sqi[:2])
                sample_healthy_fs = fs_rec # Update fs
        
        if len(c_cbfv) > 0 and len(sample_corrupted) < 5:
             sample_corrupted.extend(c_cbfv[:2])
             sample_corrupted_reason.extend(c_reasons[:2])
             sample_corrupted_fs = fs_rec # Update fs for corrupted samples

        if len(v_cbfv) > 0 or len(c_cbfv) > 0:
            save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segments.npz")
            np.savez(save_path, 
                     valid_cbfv=v_cbfv, 
                     valid_sqi=v_sqi, 
                     corrupted_cbfv=c_cbfv, 
                     corrupted_reasons=c_reasons,
                     class_label='Healthy',
                     fs=fs_rec) # Save fs too
            print(f"  Saved {len(v_cbfv)} valid segments to {save_path}")
        else:
            print(f"  No segments generated for {filename}")
        
    # Process ICU
    for f in icu_files:
        filename = os.path.basename(f)
        print(f"Processing {filename} from {icu_data_path}...")
        v_cbfv, v_sqi, c_cbfv, c_reasons, fs_rec = process_recording(f)
        
        if len(v_cbfv) > 0:
             if len(sample_icu_valid) < 5:
                sample_icu_valid.extend(v_cbfv[:2])
                sample_icu_sqi.extend(v_sqi[:2])
                sample_icu_fs = fs_rec # Update fs
                
        if len(c_cbfv) > 0 and len(sample_corrupted) < 10: # More corrupted samples
             sample_corrupted.extend(c_cbfv[:2])
             sample_corrupted_reason.extend(c_reasons[:2])
             sample_corrupted_fs = fs_rec # Update fs for corrupted samples
        
        if len(v_cbfv) > 0 or len(c_cbfv) > 0:
            save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segments.npz")
            np.savez(save_path, 
                     valid_cbfv=v_cbfv, 
                     valid_sqi=v_sqi, 
                     corrupted_cbfv=c_cbfv, 
                     corrupted_reasons=c_reasons,
                     class_label='ICU',
                     fs=fs_rec)
            print(f"  Saved {len(v_cbfv)} valid segments to {save_path}")
        else:
            print(f"  No segments generated for {filename}")

    print("\n--- Processing Complete ---")
    print(f"Check the '{output_dir}' folder for individual .npz files.")

    # --- Visualization ---
    print("\n--- Visualizing Sample Segments (Envelope + SQI) ---")
    
    # 1. Visualize Valid Healthy
    if len(sample_healthy_valid) > 0:
        idx = random.randint(0, len(sample_healthy_valid) - 1)
        visualize_single_segment(
            sample_healthy_valid[idx], 
            sqi_segment=sample_healthy_sqi[idx],
            title="Valid Healthy Segment", 
            segment_idx=idx,
            fs=sample_healthy_fs
        )
        
    # 2. Visualize Valid ICU
    if len(sample_icu_valid) > 0:
        idx = random.randint(0, len(sample_icu_valid) - 1)
        visualize_single_segment(
            sample_icu_valid[idx], 
            sqi_segment=sample_icu_sqi[idx],
            title="Valid ICU Segment", 
            segment_idx=idx,
            fs=sample_icu_fs
        )

    # 3. Visualize Corrupted
    if len(sample_corrupted) > 0:
        idx = random.randint(0, len(sample_corrupted) - 1)
        reason = sample_corrupted_reason[idx]
        visualize_single_segment(
            sample_corrupted[idx], 
            title="Corrupted Segment", 
            segment_idx=idx, 
            reason=reason,
            fs=sample_corrupted_fs if sample_corrupted_fs else 217.0 # Fallback fs
        )
    else:
        print("No corrupted segments found to visualize.")

if __name__ == "__main__":
    main()
