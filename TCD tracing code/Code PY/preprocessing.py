import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
import os

def load_tcd_data(filepath):
    """
    Load TCD data from text file. 
    Supports comma-separated or tab-separated values.
    Expected columns: t, I, Q
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        # Try reading as comma separated first (based on user snippet)
        # Assuming no header if the first row looks like data
        # We'll read a few lines to sniff
        with open(filepath, 'r') as f:
            first_line = f.readline()
            
        if ',' in first_line:
            sep = ','
        else:
            sep = '\t' # fallback to tab
            
        # Check if header exists? 
        # If first line contains characters that are not numbers (except separators), it's a header.
        has_header = None
        try:
            # simple heuristic: try float conversion of first line parts
            parts = first_line.strip().split(sep)
            [float(p) for p in parts]
            has_header = None # No header
        except ValueError:
            has_header = 0 # Has header (row 0)
            
        df = pd.read_csv(filepath, sep=sep, header=has_header)
        
        if has_header is None:
            # Assign names if 3 columns
            if df.shape[1] == 3:
                df.columns = ['t', 'I', 'Q']
            else:
                # fallback, maybe just t, I, Q are first 3
                df.columns = ['t', 'I', 'Q'] + [f'col_{i}' for i in range(3, df.shape[1])]
        else:
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
        # Ensure columns exist
        required = ['t', 'I', 'Q']
        if not all(col in df.columns for col in required):
             # Try mapping by index
             if df.shape[1] >= 3:
                 df = df.iloc[:, :3]
                 df.columns = ['t', 'I', 'Q']
             else:
                 raise ValueError("Data must have at least 3 columns (t, I, Q)")
                 
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def preprocess_time_vector(t):
    """
    Preprocess the time vector to handle overflows and zeroing.
    """
    t = np.array(t, dtype=float)
    # Zero the timer and convert milliseconds to seconds (assuming input is microseconds based on /10^6)
    t = (t - t[0]) / 1e6

    # Sampling rate
    diff_t = np.diff(t)
    Ts = np.median(diff_t)

    # Correct for overflow in time vector (detect negative jumps in time stamps)
    overflow_index = np.where(diff_t < -Ts)[0]
    
    current_t = t.copy()
    
    for idx in overflow_index:
        delta_t = np.abs(current_t[idx] - current_t[idx+1]) + Ts
        current_t[idx+1:] += delta_t
        
    return current_t

def segment_echo_signal(I, Q, t, window_length=60.0):
    """
    Segment received echo signal (IQ) into segments of specified duration.
    
    Parameters:
    - I, Q: Signal components
    - t: Time vector
    - window_length: Duration of each segment in seconds (default: 60.0)
    """
    t = np.array(t)
    I = np.array(I)
    Q = np.array(Q)
    
    # Initializations
    t_list = []
    IQ_list = []
    fs_list = []
    ix_low_freq_list = []
    
    # Find jumps > 0.5s
    diff_t = np.diff(t)
    jump_indices = np.where(diff_t > 0.5)[0]
    
    boundaries = [0] + list(jump_indices + 1) + [len(t)]
    
    ix_segments = []
    
    # Get segments
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        if start_idx >= end_idx:
            continue
            
        ix_ta = start_idx
        
        while True:
            t_end_block = t[end_idx - 1]
            t_start = t[ix_ta]
            
            # Use the window_length argument here
            if t_start + window_length + window_length - 5 > t_end_block:
                tb = t_end_block
                win_end_reached = True
            else:
                tb = t_start + window_length
                win_end_reached = False
            
            search_slice = t[ix_ta:end_idx]
            local_idx = np.argmin(np.abs(search_slice - tb))
            ix_tb = ix_ta + local_idx
            
            ix_segments.append((ix_ta, ix_tb))
            
            if win_end_reached:
                break
            else:
                ix_ta = ix_tb + 1
                if ix_ta >= end_idx:
                    break

    # Populate lists
    for (seg_start, seg_end) in ix_segments:
        t_seg = t[seg_start : seg_end + 1]
        I_seg = I[seg_start : seg_end + 1]
        Q_seg = Q[seg_start : seg_end + 1]
        
        t_tmp, unique_indices = np.unique(t_seg, return_index=True)
        I_tmp = I_seg[unique_indices]
        Q_tmp = Q_seg[unique_indices]
        
        diff_t_tmp = np.diff(t_tmp)
        if len(diff_t_tmp) > 0:
            median_diff = np.median(diff_t_tmp)
            ix_low_freq = np.where(diff_t_tmp > 1.1 * median_diff)[0]
            ix_low_freq_list.append(ix_low_freq)
        else:
            ix_low_freq_list.append(np.array([]))
            
        # Resample
        Ts = 1.4400e-04
        fs = 1/Ts
        
        if len(t_tmp) > 1:
            t_uniform = np.arange(t_tmp[0], t_tmp[-1], Ts)
            
            if len(t_uniform) > 0:
                interp_I = PchipInterpolator(t_tmp, I_tmp)(t_uniform)
                interp_Q = PchipInterpolator(t_tmp, Q_tmp)(t_uniform)
                
                IQ_complex = interp_I + 1j * interp_Q
                
                t_list.append(t_uniform)
                IQ_list.append(IQ_complex)
                fs_list.append(fs)
            else:
                pass

    return {
        't': t_list,
        'IQ': IQ_list,
        'fs': fs_list,
        'ix_low_freq': ix_low_freq_list
    }

def segment_cbfv_into_windows(cbfv_signal, window_length=1024, overlap=0):
    """
    Segments a 1D CBFV signal into fixed-length windows with an optional overlap.

    Parameters:
    - cbfv_signal (np.ndarray): The 1D CBFV envelope signal.
    - window_length (int): The desired length of each segment (number of samples).
    - overlap (int): The number of samples to overlap between consecutive windows.

    Returns:
    - List[np.ndarray]: A list of numpy arrays, where each array is a segment.
    """
    segments = []
    num_samples = len(cbfv_signal)
    step_size = window_length - overlap

    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if overlap >= window_length or overlap < 0:
        raise ValueError("overlap must be less than window_length and non-negative.")

    start = 0
    while start + window_length <= num_samples:
        segment = cbfv_signal[start : start + window_length]
        segments.append(segment)
        start += step_size
    
    # Optional: handle remaining partial segment if needed. For now, we discard.
    # If a partial segment at the end is desired, add logic here.
    # For anomaly detection, full segments are usually preferred.

    return segments

