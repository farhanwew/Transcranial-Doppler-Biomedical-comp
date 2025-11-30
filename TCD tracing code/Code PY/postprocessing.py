import numpy as np
from scipy import signal

def postprocess_cbfv(cbfv, fs):
    if fs is None or np.isnan(fs):
        return cbfv
        
    fc = 50
    f_nyquist = fs / 2
    order = 2
    
    # LPF
    b, a = signal.butter(order, fc / f_nyquist, 'low') # Matlab default is low? butter(n, Wn) is low pass.
    cbfv_filt = signal.filtfilt(b, a, cbfv)
    
    n = int(round(0.05 * fs))
    if n % 2 == 0: n += 1
    
    cbfv_final = signal.medfilt(cbfv_filt, kernel_size=n)
    
    return cbfv_final
