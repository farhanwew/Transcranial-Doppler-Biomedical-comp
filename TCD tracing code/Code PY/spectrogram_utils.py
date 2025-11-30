import numpy as np
from scipy import signal

def compute_spectrogram(IQ, fwc, fs, nsamples, fracoverlap, NFFT):
    """
    Compute spectrogram from complex IQ data.
    
    Parameters:
    - IQ: Complex IQ data
    - fwc: Wall filter cutoff frequency (Hz)
    - fs: Sampling frequency
    - nsamples: Window size
    - fracoverlap: Fraction of overlap (0 to 1)
    - NFFT: FFT length
    """
    
    # Filter IQ data (Highpass)
    nyquist = fs / 2
    b, a = signal.butter(2, fwc / nyquist, 'high')
    
    # filtfilt requires real coefficients, which we have.
    # IQ is complex. filtfilt handles complex data by filtering real and imag parts separately? 
    # Scipy documentation says it handles complex input.
    IQ_filtered = signal.filtfilt(b, a, IQ)
    
    IQ_length = len(IQ_filtered)
    # Truncate to multiple of nsamples
    limit = (IQ_length // nsamples) * nsamples
    IQ_filtered = IQ_filtered[:limit]
    
    # Generate spectrogram
    # Matlab: spectrogram(x, window, noverlap, nfft, fs, 'centered')
    # Matlab 'centered' puts 0 frequency in the middle.
    
    window = signal.get_window('hann', nsamples)
    noverlap = int(fracoverlap * nsamples)
    
    # scipy.signal.spectrogram returns (f, t, Sxx)
    # return_onesided=False is needed for complex data to get full spectrum.
    # scaling='spectrum' ? Matlab returns Power Spectral Density (PSD) by default in the 4th argument (P).
    # Actually Matlab's 4th arg P is the PSD. S is the STFT.
    # The code uses SP (the 4th arg).
    
    f, t, Sxx = signal.spectrogram(
        IQ_filtered, 
        fs=fs, 
        window=window, 
        nperseg=nsamples, 
        noverlap=noverlap, 
        nfft=NFFT, 
        detrend=False, 
        return_onesided=False, 
        scaling='spectrum', # 'density' is V**2/Hz, 'spectrum' is V**2
        mode='psd' 
    )
    
    # Shift zero frequency to center
    f_shifted = np.fft.fftshift(f)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    
    # S is not returned in the Python translation as it wasn't used in the calling code significantly, 
    # but the signature asks for it. Matlab's S is STFT (complex). 
    # If we need it, we'd use mode='complex'. 
    # But the main output used is SP (Sxx).
    
    return None, f_shifted, t, Sxx_shifted

def generate_spectrogram(IQ, fs, t):
    """
    Generate Doppler spectrogram.
    """
    nsamples = 128
    frac_overlap = 0.75
    NFFT = 1024
    fwc = 100 # Hz
    
    _, f_spec, t_spec, SP = compute_spectrogram(IQ, fwc, fs, nsamples, frac_overlap, NFFT)
    
    # fs_Spec = 1/median(diff(tSpectrogram))
    if len(t_spec) > 1:
        fs_spec = 1.0 / np.median(np.diff(t_spec))
    else:
        fs_spec = 0 # Handle edge case
        
    t_spec = t_spec + t[0]
    
    return f_spec, t_spec, SP, fs_spec
