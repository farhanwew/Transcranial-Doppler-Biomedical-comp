import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Import local modules
from preprocessing import preprocess_time_vector, segment_echo_signal, load_tcd_data, segment_cbfv_into_windows
from spectrogram_utils import generate_spectrogram
from tracing import spectrogram_tracing, ultrasound_tracing_mtcm
from postprocessing import postprocess_cbfv

def main():
    print("Starting TCD Trace Processing...")
    
    # Options
    run_adaptive_env_detection = True
    run_mtcm_env_detection = True
    
    # Parameters
    fc = 1.75e6
    ss = 1540
    angle = 0
    size_sp_filt = [3, 3] # Median filter kernel
    
    # Initializations
    cbfv_mtcm = []
    cbfv_adaptive = []
    cbfv_adaptive_raw = [] # To store the adaptive envelope before post-processing
    cbfv_sqi = []
    t_spectrogram_all = [0]
    sp_bnw_list = [] # To aggregate binarized spectrograms
    sp_list = []
    
    # Path to recordings
    # Adjust path as necessary
    filepath = '../Healthy Subjects/' 
    filename = 'Healthy_Subjects_Recording_1.txt'
    full_path = os.path.join(filepath, filename)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        # Check standard locations
        possible_paths = [
            os.path.join('TCD tracing code', 'Healthy Subjects', filename),
            filename
        ]
        for p in possible_paths:
            if os.path.exists(p):
                full_path = p
                break
    
    if os.path.exists(full_path):
        print(f"Loading data from {full_path}...")
        try:
            raw_echo_data = load_tcd_data(full_path)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
    else:
        print(f"File still not found. Generating dummy data for demonstration...")
        t_dummy = np.arange(0, 10, 1/10000.0) # 10 seconds, 10kHz
        # Create a dummy signal: IQ
        # Doppler shift around 1kHz -> ~40cm/s
        carrier = np.exp(1j * 2 * np.pi * 1000 * t_dummy)
        noise = (np.random.randn(len(t_dummy)) + 1j * np.random.randn(len(t_dummy))) * 0.1
        iq_dummy_real = np.real(carrier + noise)
        iq_dummy_imag = np.imag(carrier + noise)
        
        # Create a dataframe mimicking the input
        raw_echo_data = pd.DataFrame({
            't': t_dummy * 1e6, # Convert to microseconds as expected by preprocess
            'I': iq_dummy_real,
            'Q': iq_dummy_imag
        })

    # Preprocess time
    t = preprocess_time_vector(raw_echo_data['t'].values)
    
    # Segment echo data
    tcd_echo_data = segment_echo_signal(raw_echo_data['I'].values, raw_echo_data['Q'].values, t)
    
    num_segments = len(tcd_echo_data['t'])
    print(f"Processing {num_segments} segments...")
    
    v_spectrogram = None # Will be set in loop
    
    for i in range(num_segments):
        iq_signal = tcd_echo_data['IQ'][i]
        fs_echo = tcd_echo_data['fs'][i]
        t_echo = tcd_echo_data['t'][i]
        
        # Generate Spectrogram
        f_sp, t_sp, sp_seg, fs_sp = generate_spectrogram(iq_signal, fs_echo, t_echo)
        
        # Update global time vector
        t_spectrogram_all.extend(t_sp)
        
        # Exclude indices
        ix_exclude_otsu = []
        if len(tcd_echo_data['ix_low_freq'][i]) > 0:
            # Mapping time indices to spectrogram indices
            # ix_low_freq is index in original time vector.
            # We need to map to spectrogram time axis.
            ratio = fs_sp / fs_echo
            ix_exclude_otsu = np.round(tcd_echo_data['ix_low_freq'][i] * ratio).astype(int)
            
        # Frequency to Velocity
        # v = c * f / (2 * f0 * cos(angle))
        # Factor 100 for cm/s ? Matlab code: vSpectrogram = 100*freqs_SP*ss/(2*fc*cosd(angle));
        v_spectrogram = 100 * f_sp * ss / (2 * fc * np.cos(np.deg2rad(angle)))
        
        # Doppler Spectrum Processing
        # Log compression
        sp_seg = np.log2(sp_seg + 1e-9) # Add epsilon
        max_val = np.max(sp_seg)
        if max_val != 0:
            sp_seg = sp_seg / max_val
            
        # Median Filter
        sp_seg = signal.medfilt2d(sp_seg, kernel_size=size_sp_filt)
        
        # Store
        sp_list.append(sp_seg)
        
        current_len = sp_seg.shape[1]
        
        # Adaptive Envelope
        if run_adaptive_env_detection:
            cbfv_curr_raw, cbfv_sqi_curr, img_bw_curr, _ = spectrogram_tracing(v_spectrogram, sp_seg, fs_sp, ix_exclude_otsu)
            cbfv_adaptive_raw.extend(cbfv_curr_raw) # Store raw envelope
            cbfv_curr = postprocess_cbfv(cbfv_curr_raw, fs_sp) # Post-process for filtered envelope
            
            cbfv_adaptive.extend(cbfv_curr) # Store filtered envelope
            cbfv_sqi.extend(cbfv_sqi_curr)
            sp_bnw_list.append(img_bw_curr) # Store binarized spectrogram
            
        # MTCM Envelope
        if run_mtcm_env_detection:
            cbfv_mtcm_curr = ultrasound_tracing_mtcm(v_spectrogram, sp_seg)
            cbfv_mtcm_curr = postprocess_cbfv(cbfv_mtcm_curr, fs_sp)
            cbfv_mtcm.extend(cbfv_mtcm_curr)
            
    # Handling Jumps (Add blanks)
    # Python list to numpy array
    t_spectrogram_all = np.array(t_spectrogram_all[1:]) # Remove initial 0
    
    if len(t_spectrogram_all) == 0:
        print("No data processed.")
        return

    cbfv_adaptive = np.array(cbfv_adaptive)
    cbfv_adaptive_raw = np.array(cbfv_adaptive_raw) # Convert to numpy array
    cbfv_mtcm = np.array(cbfv_mtcm)
    cbfv_sqi = np.array(cbfv_sqi)
    
    # Concatenate spectrograms
    if len(sp_list) > 0:
        SP_full = np.hstack(sp_list)
    else:
        SP_full = np.zeros((len(v_spectrogram), len(t_spectrogram_all)))

    # Concatenate binarized spectrograms
    SP_BNW_full = None
    if run_adaptive_env_detection and len(sp_bnw_list) > 0:
        SP_BNW_full = np.hstack(sp_bnw_list)
    else:
        # Fallback if no adaptive detection or empty list, match shape of SP_full
        SP_BNW_full = np.full(SP_full.shape, np.nan) 

    # Detect jumps
    diff_t = np.diff(t_spectrogram_all)
    ts = np.median(diff_t)
    ix_t_jump = np.where(diff_t > 0.25)[0]
    
    # Insert NaNs logic - tricky with arrays.
    # Simpler to reconstruct lists.
    
    if len(ix_t_jump) > 0:
        print(f"Found {len(ix_t_jump)} time jumps. Inserting NaNs...")
        # We will rebuild the arrays
        new_t = []
        new_adaptive = []
        new_adaptive_raw = [] # For raw adaptive envelope
        new_mtcm = []
        new_sqi = []
        new_sp_cols = []
        new_sp_bnw_cols = [] # For binarized spectrogram
        
        curr_idx = 0
        for jump_idx in ix_t_jump:
            # Append segment up to jump
            # jump_idx is the index BEFORE the jump.
            end_idx = jump_idx + 1
            
            new_t.extend(t_spectrogram_all[curr_idx:end_idx])
            if run_adaptive_env_detection: 
                new_adaptive.extend(cbfv_adaptive[curr_idx:end_idx])
                new_adaptive_raw.extend(cbfv_adaptive_raw[curr_idx:end_idx])
                new_sqi.extend(cbfv_sqi[curr_idx:end_idx])
                new_sp_bnw_cols.append(SP_BNW_full[:, curr_idx:end_idx]) # Append B&W segment
            if run_mtcm_env_detection: new_mtcm.extend(cbfv_mtcm[curr_idx:end_idx])
            new_sp_cols.append(SP_full[:, curr_idx:end_idx])
            
            # Create gap
            t_pre = t_spectrogram_all[jump_idx]
            t_post = t_spectrogram_all[jump_idx+1]
            
            gap_t = np.arange(t_pre + ts, t_post, ts)
            gap_len = len(gap_t)
            
            new_t.extend(gap_t)
            if run_adaptive_env_detection: 
                new_adaptive.extend([np.nan] * gap_len)
                new_adaptive_raw.extend([np.nan] * gap_len)
                new_sqi.extend([np.nan] * gap_len)
                new_sp_bnw_cols.append(np.full((SP_full.shape[0], gap_len), np.nan)) # Append NaN for B&W
            if run_mtcm_env_detection: new_mtcm.extend([np.nan] * gap_len)
            new_sp_cols.append(np.full((SP_full.shape[0], gap_len), np.nan))
            
            curr_idx = end_idx
            
        # Append remaining
        new_t.extend(t_spectrogram_all[curr_idx:])
        if run_adaptive_env_detection: 
            new_adaptive.extend(cbfv_adaptive[curr_idx:])
            new_adaptive_raw.extend(cbfv_adaptive_raw[curr_idx:])
            new_sqi.extend(cbfv_sqi[curr_idx:])
            new_sp_bnw_cols.append(SP_BNW_full[:, curr_idx:]) # Append remaining B&W
        if run_mtcm_env_detection: new_mtcm.extend(cbfv_mtcm[curr_idx:])
        new_sp_cols.append(SP_full[:, curr_idx:])
        
        t_spectrogram_all = np.array(new_t)
        cbfv_adaptive = np.array(new_adaptive)
        cbfv_adaptive_raw = np.array(new_adaptive_raw) # Convert to numpy array
        cbfv_mtcm = np.array(new_mtcm)
        cbfv_sqi = np.array(new_sqi)
        SP_full = np.hstack(new_sp_cols)
        SP_BNW_full = np.hstack(new_sp_bnw_cols) # Convert to numpy array
        

    # Plotting
    print("Plotting results...")
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(10, 14)) # 5 subplots now
    
    # Extent [left, right, bottom, top]
    extent = [t_spectrogram_all[0], t_spectrogram_all[-1], v_spectrogram[0], v_spectrogram[-1]]
    
    # 1. Spectrogram with Traces
    ax[0].imshow(SP_full, aspect='auto', origin='lower', extent=extent, cmap='gray', vmin=0.1, vmax=0.95)
    if run_adaptive_env_detection:
        ax[0].plot(t_spectrogram_all, cbfv_adaptive, 'r', linewidth=1.5, label='Adaptive') # Use filtered for main plot
    if run_mtcm_env_detection:
        ax[0].plot(t_spectrogram_all, cbfv_mtcm, 'g', linewidth=1.5, label='MTCM') # Changed to green
    ax[0].set_ylabel('CBFV [cm/s]')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Automatic envelope tracing by proposed algorithm (red) and MTCM (green)')
    
    # 2. Binarized Spectrogram
    ax[1].imshow(SP_BNW_full, aspect='auto', origin='lower', extent=extent, cmap='gray', vmin=0.0, vmax=1.0)
    ax[1].set_ylabel('CBFV [cm/s]')
    ax[1].set_title('Binarized spectrogram')
    
    # 3. Raw Adaptive Envelope
    if run_adaptive_env_detection:
        ax[2].plot(t_spectrogram_all, cbfv_adaptive_raw, 'r', linewidth=1.5, label='Raw Envelope')
    ax[2].set_ylabel('CBFV [cm/s]')
    ax[2].legend(loc='upper right')
    ax[2].set_title('Raw envelope signal with some spikes')
    
    # 4. Filtered Adaptive Envelope
    if run_adaptive_env_detection:
        ax[3].plot(t_spectrogram_all, cbfv_adaptive, 'r', linewidth=1.5, label='Filtered Envelope')
    ax[3].set_ylabel('CBFV [cm/s]')
    ax[3].legend(loc='upper right')
    ax[3].set_title('Filtered final output')
    
    # 5. SQI
    if run_adaptive_env_detection:
        ax[4].plot(t_spectrogram_all, 100 * cbfv_sqi, 'r', linewidth=1.5)
    ax[4].set_ylabel('SQI [%]')
    ax[4].set_xlabel('Time [s]')
    ax[4].set_title('Signal quality index')
    
    plt.tight_layout()
    plt.savefig('TCD_Trace_Result.png')
    print("Results saved to TCD_Trace_Result.png")
    # plt.show() 

    # --- Demonstrating new CBFV windowing function ---
    print("\n--- Segmenting CBFV into 1024-sample windows ---")
    if run_adaptive_env_detection and len(cbfv_adaptive) > 0:
        cbfv_windows = segment_cbfv_into_windows(cbfv_adaptive, window_length=1024, overlap=0)
        print(f"Created {len(cbfv_windows)} CBFV windows of length 1024.")
        # You can now further process cbfv_windows for VAE or other analysis
    else:
        print("No adaptive CBFV data to segment.")


if __name__ == "__main__":
    main()
