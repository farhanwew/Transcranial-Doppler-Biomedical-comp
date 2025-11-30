import numpy as np
from scipy import signal

def wabp(abp):
    """
    WABP - ABP waveform onset detector.
    Adapted from Zong's wabp.c and Matlab implementation.
    """
    if abp.ndim != 1:
        raise ValueError("Input must be a 1D array")

    # Scale physiologic ABP
    offset = 1600
    scale = 20
    Araw = abp * scale - offset

    # Low Pass Filter
    # A = filter([1 0 0 0 0 -2 0 0 0 0 1],[1 -2 1],Araw)/24+30;
    b = np.array([1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 1])
    a = np.array([1, -2, 1])
    
    # signal.lfilter corresponds to Matlab's filter
    A = signal.lfilter(b, a, Araw) / 24 + 30
    
    # Takes care of 4 sample group delay
    # A = (A(4:end)+offset)/scale; 
    # In Python A[3:] but we need to be careful about array length.
    if len(A) > 3:
        A = (A[3:] + offset) / scale
    else:
        return np.array([])

    # Slope-sum function
    dypos = np.diff(A)
    dypos[dypos < 0] = 0
    
    # ssf = [0; 0; conv(ones(16,1),dypos)];
    # conv(..., 'full') by default in Matlab.
    # ones(16,1) is a window of 16 ones.
    
    kernel = np.ones(16)
    # We use 'valid' or 'full'? Matlab conv(u,v) is full.
    # dypos length is N-1. Kernel 16. Result N+14.
    # Matlab code prepends two zeros.
    conv_res = np.convolve(dypos, kernel) # default is full
    ssf = np.concatenate(([0, 0], conv_res))
    
    # Decision rule
    if len(ssf) < 1000:
        avg0 = np.mean(ssf)
    else:
        avg0 = np.mean(ssf[:1000])
    
    threshold0 = 3 * avg0
    
    lockout = 0
    timer = 0
    z = [] # Detected onsets
    
    # Loop
    # for t = 50:length(ssf)-17
    # Matlab indices 1-based. So t starts at 50.
    # Python 0-based. t starts 49.
    
    limit = len(ssf) - 17
    
    # We need to maintain state variables
    
    for t in range(49, limit):
        lockout -= 1
        timer += 1
        
        if lockout < 1 and ssf[t] > avg0 + 5:
            timer = 0
            # maxSSF = max(ssf(t:t+16)); Matlab t:t+16 is 17 samples? 
            # No, t:t+16 includes t and t+16. Length 17.
            # Python t:t+17
            
            max_ssf_window = ssf[t : t+17]
            max_ssf = np.max(max_ssf_window)
            
            # minSSF = min(ssf(t-16:t)); 
            min_ssf_window = ssf[t-16 : t+1] # Includes t
            min_ssf = np.min(min_ssf_window)
            
            if max_ssf > (min_ssf + 10):
                onset = 0.01 * max_ssf
                
                # tt = t-16:t
                tt_indices = np.arange(t-16, t+1)
                dssf = ssf[tt_indices] - ssf[tt_indices - 1]
                
                # BeatTime = find(dssf<onset,1,'last')+t-17;
                # find last index where dssf < onset
                candidates = np.where(dssf < onset)[0]
                
                if len(candidates) > 0:
                    last_idx = candidates[-1]
                    # Mapping back to global index
                    # The local index 'last_idx' corresponds to tt_indices
                    # Wait, Matlab: find returns index relative to dssf array.
                    # dssf has length 17.
                    # if last_idx is 1 (1-based), it means dssf(1).
                    # dssf(1) corresponds to ssf(t-16) - ssf(t-17).
                    # BeatTime = last_idx + t - 17.
                    # Python: last_idx (0-based).
                    # BeatTime = last_idx + t - 16?
                    # Let's check logic. 
                    # t-17 in Matlab is offset.
                    # If candidates[-1] is 0. It maps to t-16.
                    # t-16 = 0 + t - 16. 
                    # So BeatTime = last_idx + (t - 16).
                    
                    beat_time = last_idx + (t - 16)
                    z.append(beat_time)
                    
                    threshold0 = threshold0 + 0.1 * (max_ssf - threshold0)
                    avg0 = threshold0 / 3
                    lockout = 32
        
        if timer > 312:
            threshold0 = threshold0 - 1
            avg0 = threshold0 / 3
            
    # r = z(find(z))-2;
    # -2 shift
    r = np.array(z) - 2
    return r.astype(int)


def wabp_wrapper(cbfv, fs):
    beat_onsets = []
    
    # Resample to 125 Hz
    fs_new = 125.0
    num_samples = int(len(cbfv) * fs_new / fs)
    cbfv_resampled = signal.resample(cbfv, num_samples)
    
    # Find median peak height to check if signal is flat/noise
    # MinPeakHeight 20, MinPeakDistance 0.5*fsNew
    peaks, properties = signal.find_peaks(cbfv_resampled, height=20, distance=0.5*fs_new)
    
    if len(peaks) == 0:
        return np.array([])
        
    # Find beat onsets
    # wabp expects input scaled/normalized? 
    # wabp(140*cbfvResampled/median(cbfv_pks));
    peak_heights = properties['peak_heights']
    median_peak = np.median(peak_heights)
    
    if median_peak == 0:
        return np.array([])
        
    scaled_input = 140 * cbfv_resampled / median_peak
    beat_onsets_resampled = wabp(scaled_input)
    
    if len(beat_onsets_resampled) == 0:
        return np.array([])
        
    # Transform back
    beat_onsets = np.round(beat_onsets_resampled * (fs / fs_new)).astype(int)
    
    # Small realignment
    win_size = int(round(0.05 * fs))
    len_signal = len(cbfv)
    
    corrected_onsets = []
    for onset in beat_onsets:
        lb = max(0, onset - win_size)
        rb = min(len_signal, onset + win_size) # Slice exclusive?
        
        if lb >= rb:
            corrected_onsets.append(onset)
            continue
            
        segment = cbfv[lb:rb]
        ix_corrected = np.argmin(segment)
        corrected_onsets.append(lb + ix_corrected) # 0-based index logic holds
        
    return np.array(corrected_onsets)

def amplitude_based_labeling(cbfv, beat_onsets):
    is_artifact = np.zeros(len(cbfv))
    min_pulsatility = 20
    min_amplitude = 30
    
    if len(beat_onsets) == 0:
        is_artifact[:] = 1
        return is_artifact
        
    for i in range(len(beat_onsets) - 1):
        start = beat_onsets[i]
        end = beat_onsets[i+1] # Exclusive
        
        window_slice = cbfv[start:end]
        if len(window_slice) == 0:
            continue
            
        min_cbfv = np.min(window_slice)
        max_cbfv = np.max(window_slice)
        
        if max_cbfv < min_amplitude:
            is_artifact[start:end] = 1
            continue
            
        pulsatility = max_cbfv - min_cbfv
        
        if pulsatility < min_pulsatility and max_cbfv > 45:
            is_artifact[start:end] = 1
            continue
            
        if pulsatility < 0.5 * np.mean(window_slice):
            is_artifact[start:end] = 1
            
    return is_artifact

def period_based_labeling(cbfv, beat_onsets, binary_label_amp, fs):
    min_duration = 0.25
    max_duration = 2.0
    
    diff_onsets = np.diff(beat_onsets)
    if len(diff_onsets) == 0:
        return np.ones(len(cbfv)), np.array([])
        
    median_period = np.median(diff_onsets) / fs # Convert to seconds if diff is samples? 
    # Matlab code: medianPeriod = median(diffOnsets); (This is in samples)
    # Then minPeriod = 0.3 * medianPeriod; (Samples)
    
    median_period_samples = np.median(diff_onsets)
    min_period = 0.3 * median_period_samples
    max_period = 3.0 * median_period_samples
    
    binary_label = np.zeros(len(cbfv))
    pulse_collection = []
    
    for i in range(len(beat_onsets) - 1):
        start = beat_onsets[i]
        end = beat_onsets[i+1]
        
        pulse_window = cbfv[start:end+1] # Matlab includes end? beatOnsets(i):beatOnsets(i+1)
        # Wait, beatOnsets(i) to beatOnsets(i+1). 
        # Overlaps at boundary? 
        # Usually beatOnsets(i):beatOnsets(i+1)-1 is a beat.
        # Matlab code says: pulseWindow = beatOnsets(i):beatOnsets(i+1); 
        # This implies one sample overlap or inclusive end.
        
        amp_artifact = binary_label_amp[start]
        
        pulse_duration_s = diff_onsets[i] / fs
        abs_duration_bad = pulse_duration_s < min_duration or pulse_duration_s > max_duration
        rel_duration_bad = diff_onsets[i] < min_period or diff_onsets[i] > max_period
        
        if abs_duration_bad or rel_duration_bad or amp_artifact != 0:
            binary_label[start : end+1] = 1 # Mark segment
        else:
            pulse_collection.append(pulse_window)
            
    if len(pulse_collection) > 0:
        # Median template needs pulses of same length? 
        # Matlab code: pulse_collection = zeros(numBeats, max(diffOnsets));
        # It pads with zeros? No, it initializes with max length.
        # Then takes median column-wise.
        
        max_len = 0
        for p in pulse_collection:
            max_len = max(max_len, len(p))
            
        # Pad with NaNs or extend? 
        # Matlab: pulse_collection(pulseCnt, 1:pulse_length) = pulseWave;
        # The rest remains 0.
        
        mat = np.zeros((len(pulse_collection), max_len))
        for idx, p in enumerate(pulse_collection):
            mat[idx, :len(p)] = p
            
        median_template = np.median(mat, axis=0)
    else:
        median_template = np.array([])
        
    return binary_label, median_template

def nrms_classification(template_pulse, candidate_pulse, fs):
    max_nrmse = 0.25
    
    # Normalize
    template_pulse = template_pulse - template_pulse[0]
    if np.max(template_pulse) != 0:
        template_pulse /= np.max(template_pulse)
        
    candidate_pulse = candidate_pulse - candidate_pulse[0]
    if np.max(candidate_pulse) != 0:
        candidate_pulse /= np.max(candidate_pulse)
        
    # Pad candidate if shorter
    if len(template_pulse) > len(candidate_pulse):
        pad_len = len(template_pulse) - len(candidate_pulse)
        candidate_pulse = np.pad(candidate_pulse, (0, pad_len), 'edge')
        
    # Cross-correlation alignment
    # maxlag = round(0.15*fs)
    max_lag = int(round(0.15 * fs))
    
    # scipy correlate 'full'. 
    # We want lags around 0.
    correlation = signal.correlate(template_pulse, candidate_pulse, mode='full')
    lags = signal.correlation_lags(len(template_pulse), len(candidate_pulse), mode='full')
    
    # Restrict to max_lag
    mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_lags = lags[mask]
    valid_corr = correlation[mask]
    
    if len(valid_corr) > 0:
        best_idx = np.argmax(valid_corr)
        lag_best = valid_lags[best_idx]
    else:
        lag_best = 0
        
    # Align
    if lag_best == 0:
        tp = template_pulse
        cp = candidate_pulse
    elif lag_best < 0:
        # template moves left relative to candidate? 
        # Matlab: templatePulse = templatePulse(1:end+lagBest); (lagBest is negative)
        # candidatePulse = candidatePulse(-lagBest:end);
        
        # Python equivalent
        tp = template_pulse[:lag_best] # removes last |lagBest|
        cp = candidate_pulse[-lag_best:] # removes first |lagBest|
    else:
        tp = template_pulse[lag_best:]
        cp = candidate_pulse[:-lag_best]
        
    # Window MSE
    limit = int(round(0.75 * len(template_pulse)))
    # Ensure we don't go out of bounds of aligned arrays
    limit = min(limit, len(tp), len(cp))
    
    error_sig = tp[:limit] - cp[:limit]
    
    denom = np.sum(tp[:limit]**2)
    if denom == 0:
        nrmse = 1.0
    else:
        nrmse = np.sqrt(min(1, np.sum(error_sig**2) / denom))
        
    artifact = 0 if nrmse < max_nrmse else 1
    sqi = 1 - nrmse
    
    return artifact, sqi

def template_based_labeling(cbfv, beat_onsets, artifact_label, median_template, fs):
    len_cbfv = len(cbfv)
    is_artifact = np.ones(len_cbfv)
    sqi_signal = np.zeros(len_cbfv)
    
    if len(beat_onsets) < 2:
        return is_artifact, sqi_signal
        
    sqi_signal[:beat_onsets[0]] = np.nan
    sqi_signal[beat_onsets[-1]:] = np.nan
    
    # Clean template
    # medianTemplate(medianTemplate < 10) = [];
    median_template = median_template[median_template >= 10]
    
    if len(median_template) == 0:
        return is_artifact, sqi_signal
        
    pulsatility = np.max(median_template) - np.min(median_template)
    if pulsatility < 25 or pulsatility < 0.5 * np.mean(median_template):
        return is_artifact, sqi_signal
        
    # Loop
    for i in range(len(beat_onsets) - 1):
        start = beat_onsets[i]
        end = beat_onsets[i+1] - 1 # Matlab: beatOnsets(i+1)-1
        
        if start >= end:
            continue
            
        pulse_window_indices = np.arange(start, end+1)
        pulse_wave = cbfv[pulse_window_indices]
        
        # artifactLabel(pulseWindow(2:end-1))
        # Python indices 1:-1 relative to window
        inner_indices = pulse_window_indices[1:-1]
        if len(inner_indices) > 0:
            amp_or_period_artifact = np.sum(artifact_label[inner_indices])
        else:
            amp_or_period_artifact = 0
            
        if amp_or_period_artifact == 0:
            artifact_mse, sqi = nrms_classification(median_template, pulse_wave, fs)
            sqi_signal[pulse_window_indices] = sqi
            
            if artifact_mse == 0:
                is_artifact[pulse_window_indices] = 0
                
    return is_artifact, sqi_signal

def sqa_cbfv(cbfv, fs):
    beat_onsets = wabp_wrapper(cbfv, fs)
    seg_duration = len(cbfv) / (60.0 * fs) # minutes
    
    if len(beat_onsets) / seg_duration < 10 or np.median(cbfv) > 120:
        good_sig_percent = 0
        sqi_signal = np.zeros(len(cbfv))
    else:
        binary_label_amp = amplitude_based_labeling(cbfv, beat_onsets)
        binary_label_period, cbfv_template = period_based_labeling(cbfv, beat_onsets, binary_label_amp, fs)
        binary_label_model, sqi_signal = template_based_labeling(cbfv, beat_onsets, binary_label_period, cbfv_template, fs)
        
        artifacts = np.nonzero(binary_label_model)[0]
        good_sig_percent = 1 - len(artifacts) / len(cbfv)
        
    return good_sig_percent, sqi_signal
