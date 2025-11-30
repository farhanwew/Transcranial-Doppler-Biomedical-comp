import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import signal

# Import local modules
from preprocessing import preprocess_time_vector, load_tcd_data
from spectrogram_utils import generate_spectrogram

def main():
    print("Starting Spectrogram Visualization...")
    
    # Parameters
    fc = 1.75e6; # Transmit center frequency (carrier freq.)
    ss = 1540; # Assumed speed of sound [m/s]
    angle = 0; # Doppler angle [deg]
    size_sp_filt = [3, 3]; # Kernel size for median filter

    # Load echo and generate IQ signal
    # Adjust path as necessary
    filepath = '../Healthy Subjects/' 
    filename = 'Healthy_Subjects_Recording_1.txt'
    full_path = os.path.join(filepath, filename)
    
    # Check if file exists (fallback for user demo)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        # Try standard location if running from root
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
        df = load_tcd_data(full_path)
        t_echo = df['t'].values
        I = df['I'].values
        Q = df['Q'].values
    else:
        print("Generating dummy data...")
        t_echo = np.arange(0, 10000000, 100) # arbitrary units
        I = np.random.randn(len(t_echo))
        Q = np.random.randn(len(t_echo))

    # Preprocess Time
    # Reuse the logic from preprocessing which matches the loop
    t_echo = preprocess_time_vector(t_echo)
    
    # Resample IQ signal
    # TsEcho = 1.4400e-04; % T=1/fs such that velcity range till 152 cm/s
    ts_echo = 1.4400e-04
    fs_echo = 1.0 / ts_echo
    
    # Pchip interpolation requires unique sorted x
    t_unique, unique_indices = np.unique(t_echo, return_index=True)
    I_unique = I[unique_indices]
    Q_unique = Q[unique_indices]
    
    t_uniform = np.arange(t_unique[0], t_unique[-1], ts_echo)
    
    print("Resampling signal...")
    interp_I = PchipInterpolator(t_unique, I_unique)(t_uniform)
    interp_Q = PchipInterpolator(t_unique, Q_unique)(t_uniform)
    IQ = interp_I + 1j * interp_Q
    t_echo = t_uniform

    # Compute spectrogram
    print("Computing spectrogram...")
    f_sp, t_spectrogram, SP, fs_sp = generate_spectrogram(IQ, fs_echo, t_echo)

    # Convert frequency axis to velocity
    # vSpectrogram = 100*freqs_SP*ss/(2*fc*cosd(angle));
    v_spectrogram = 100 * f_sp * ss / (2 * fc * np.cos(np.deg2rad(angle)))

    # Doppler spectrum processing
    print("Processing spectrogram image...")
    SP = np.log2(SP + 1e-9) # Avoid log(0)
    max_val = np.max(SP)
    if max_val != 0:
        SP = SP / max_val
    
    # Median filter
    SP = signal.medfilt2d(SP, kernel_size=size_sp_filt)

    # Visualize spectrogram
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    
    # Extent [left, right, bottom, top]
    extent = [t_spectrogram[0], t_spectrogram[-1], v_spectrogram[0], v_spectrogram[-1]]
    
    plt.imshow(SP, aspect='auto', origin='lower', extent=extent, cmap='gray', vmin=0.1, vmax=0.95)
    plt.xlabel('Time [s]')
    plt.ylabel('CBFV [cm/s]')
    plt.title('TCD Spectrogram')
    plt.colorbar(label='Normalized Intensity')
    
    plt.savefig('Spectrogram_View.png')
    print("Saved Spectrogram_View.png")
    # plt.show()

if __name__ == "__main__":
    main()
