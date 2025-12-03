import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
from scipy import signal

# Import local modules
from preprocessing import preprocess_time_vector, segment_echo_signal, load_tcd_data
from spectrogram_utils import generate_spectrogram
from tracing import spectrogram_tracing
from postprocessing import postprocess_cbfv

# Parameters
FC = 1.75e6
SS = 1540
ANGLE = 0
SIZE_SP_FILT = [3, 3]

def visualize_extraction_process(data_dir='../../Healthy Subjects', output_image='extraction_process.png'):
    # 1. Find a sample file
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        # Fallback to checking current dir or ../
        files = glob.glob("*.txt") + glob.glob("../Healthy Subjects/*.txt")
    
    if not files:
        print("No data files found to visualize.")
        return

    filepath = files[0] # Just pick the first one
    print(f"Processing file: {filepath}")

    try:
        raw_data = load_tcd_data(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Preprocessing & Segmentation
    t = preprocess_time_vector(raw_data['t'].values)
    tcd_echo_data = segment_echo_signal(raw_data['I'].values, raw_data['Q'].values, t)
    
    # Pick the first valid segment
    idx = 0
    if len(tcd_echo_data['t']) > 0:
        iq_signal = tcd_echo_data['IQ'][idx]
        fs_echo = tcd_echo_data['fs'][idx]
        t_echo = tcd_echo_data['t'][idx]
        
        # 3. Spectrogram Generation
        f_sp, t_sp, sp_seg, fs_sp = generate_spectrogram(iq_signal, fs_echo, t_echo)
        
        # Velocity conversion
        v_spectrogram = 100 * f_sp * SS / (2 * FC * np.cos(np.deg2rad(ANGLE)))
        
        # Log compression & Filtering (Step 2 in description)
        sp_log = np.log2(sp_seg + 1e-9)
        max_val = np.max(sp_log)
        if max_val != 0:
            sp_log = sp_log / max_val
        sp_filt = signal.medfilt2d(sp_log, kernel_size=SIZE_SP_FILT)
        
        # 4. Tracing (Step 3 in description)
        ix_exclude_otsu = [] # Simplified
        # This tracing function returns the raw envelope and the binarized image used internally
        cbfv_raw, _, img_bw, _ = spectrogram_tracing(v_spectrogram, sp_filt, fs_sp, ix_exclude_otsu)
        
        # 5. Post-processing (Step 4 in description)
        cbfv_final = postprocess_cbfv(cbfv_raw, fs_sp)
        
        # --- PLOTTING ---
        print("Generating extraction process plot...")
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Extent [left, right, bottom, top]
        extent = [t_sp[0], t_sp[-1], v_spectrogram[0], v_spectrogram[-1]]
        
        # Panel A: Doppler Spectrogram
        axes[0].imshow(sp_filt, aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=0.4, vmax=0.95)
        axes[0].set_title("A. Preprocessed Doppler Spectrogram", fontweight='bold')
        axes[0].set_ylabel("Velocity [cm/s]")
        
        # Panel B: Binarized Spectrogram (Otsu)
        axes[1].imshow(img_bw, aspect='auto', origin='lower', extent=extent, cmap='gray')
        axes[1].set_title("B. Binarized Spectrogram (Otsu's Thresholding)", fontweight='bold')
        axes[1].set_ylabel("Velocity [cm/s]")
        
        # Panel C: Raw Envelope Tracing
        axes[2].imshow(sp_filt, aspect='auto', origin='lower', extent=extent, cmap='gray', vmin=0.1, vmax=0.95)
        axes[2].plot(t_sp, cbfv_raw, 'r', linewidth=1.5)
        axes[2].set_title("C. Adaptive Envelope Tracing (Raw)", fontweight='bold')
        axes[2].set_ylabel("Velocity [cm/s]")
        
        # Panel D: Final Filtered Waveform
        axes[3].plot(t_sp, cbfv_final, 'b', linewidth=2)
        axes[3].set_title("D. Final Post-processed CBFV Waveform", fontweight='bold')
        axes[3].set_ylabel("Velocity [cm/s]")
        axes[3].set_xlabel("Time [s]")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlim([t_sp[0], t_sp[-1]])
        axes[3].set_ylim([0, np.max(cbfv_final)*1.2]) # Auto scale Y
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        print(f"Extraction process visualization saved to {output_image}")

if __name__ == "__main__":
    # You can pass the path to your healthy data folder here
    visualize_extraction_process(data_dir='/content/TCD_Data/Healthy Subjects') 
