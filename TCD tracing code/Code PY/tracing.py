import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_otsu
from sqa import sqa_cbfv

# -------------------------------------------------------------------------
# Adaptive Method Helpers
# -------------------------------------------------------------------------

def find_contiguous_max(image_bw, velocity):
    rows, col_len = image_bw.shape
    prev_vel_env = 0
    num_miss_det = 0
    empty_max = 0
    cbfv_env = np.zeros(col_len)
    
    # Bounds for abs(CBFV)
    # velocity is increasing array? 
    # Matlab: DelV = velocity(2) - velocity(1);
    # [~, minEnv] = min(abs(velocity)); velocity = velocity(minEnv:end);
    # It seems it cuts the velocity axis to be positive/relevant part?
    # But image_bw corresponds to this? 
    # The calling code passes full image_bw? 
    # In Matlab code: 
    # if(Elower > Eupper) topSeg = imgBW_Tmp(ix_neg130:ix_neg100, :); ...
    # It seems image_bw passed here is already the slice of interest?
    # Wait, Matlab `findContiguousMax(imgBW_Tmp, vSP)` calls it with vSP (full velocity).
    # But inside: `rawBWImgCol = imageBW(minEnv:end, col);`
    # This implies imageBW matches velocity dimensions?
    # Yes, `imgBW_Tmp` is size of SP.
    
    del_v = velocity[1] - velocity[0]
    min_env_idx = np.argmin(np.abs(velocity))
    relevant_velocity = velocity[min_env_idx:]
    
    # Iterate over each time step
    for col in range(col_len):
        raw_bw_img_col = image_bw[min_env_idx:, col]
        
        # Get indices where pixel is 1
        # Matlab: find(rawBWImgCol) returns 1-based indices.
        # Python: nonzero returns tuple.
        idx_max = np.nonzero(raw_bw_img_col)[0]
        
        # Sort descend
        idx_max = np.sort(idx_max)[::-1]
        # Take top 10
        idx_max = idx_max[:min(10, len(idx_max))]
        
        if len(idx_max) == 0:
            empty_max += 1
            cbfv_env[col] = prev_vel_env
        else:
            candidate_cbfvs = relevant_velocity[idx_max]
            empty_max = 0
            
            found = False
            for i in range(len(candidate_cbfvs)):
                current_idx = idx_max[i]
                
                # 1.) Majority of pixels beneath candidate is white
                # Matlab: idxMax(i) > round(12/DelV)
                dist_idx = int(round(12.0 / del_v))
                
                if current_idx > dist_idx:
                    # mode
                    segment = raw_bw_img_col[current_idx - dist_idx : current_idx + 1]
                    # scipy.stats.mode? Or just sum > len/2?
                    # Binary image: mode is 1 if mean >= 0.5
                    if np.mean(segment) >= 0.5:
                        ones_maj = 1
                    else:
                        ones_maj = 0
                else:
                    ones_maj = 1
                    
                # 2.) Current value deviation
                if col > 4:
                    # mean(cbfv_Env(col-4:col-1))
                    prev_mean = np.mean(cbfv_env[col-4:col])
                    large_dev = (candidate_cbfvs[i] - prev_mean) > 30
                else:
                    large_dev = False
                    
                if ones_maj and not large_dev and i <= 1: # i<=2 in Matlab (1-based) -> i<=1
                    num_miss_det = 0
                else:
                    num_miss_det += 1
                    
                if (ones_maj and not large_dev) or num_miss_det > 20:
                    cbfv_env[col] = candidate_cbfvs[i]
                    prev_vel_env = candidate_cbfvs[i]
                    num_miss_det = 0
                    found = True
                    break
                    
            if not found:
                cbfv_env[col] = prev_vel_env
                
    return cbfv_env

def find_contiguous_min(image_bw, velocity):
    # Flip around imageBW and velocity? 
    # Matlab: cbfv_Env = findContiguousMax(imageBW, velocity);
    # cbfv_Env = -cbfv_Env;
    # Wait, if we pass imageBW as is, findContiguousMax looks at positive velocity logic.
    # The caller (spectrogramTracing) passes `flip(imgBW_Tmp)` and `vSP`.
    # If we flip image, the indices map to "lowest" velocities if we assume symmetry?
    # Actually, `spectrogramTracing` calls: `cbfvTmp = findContiguousMin(imgBWNeg, vSP)`
    # And `imgBWNeg` is flipped. 
    # The logic in `findContiguousMax` uses `minEnv:end`. 
    # If we want to find Min (negative envelope), we probably want to look at the "bottom" of the original image.
    # If we flip the image upside down, the bottom becomes top.
    # But `vSP` is unchanged?
    # If `vSP` ranges -150 to 150. `minEnv` is index of 0. `minEnv:end` is 0 to 150.
    # If we flip image, the data that was at -150 is now at +150 index? No.
    # Matlab `flip(imgBW_Tmp)` flips rows. Row 1 becomes Row N.
    # If Row 1 corresponds to -150 and Row N to +150.
    # After flip, Row 1 is +150 data? 
    # `findContiguousMax` ignores `velocity` VALUES except for `DelV` and returning the value.
    # It uses `velocity(idxMax)` to return value.
    # If we pass `vSP` (ordered -150 to 150), `relevant_velocity` is 0 to 150.
    # The image passed to `findContiguousMax` (via Min) is `flip(imgBW)`.
    # Original: Row 0 is -Ve Max. Row N is +Ve Max.
    # Flipped: Row 0 is +Ve Max. Row N is -Ve Max.
    # `findContiguousMax` takes `minEnv:end` (Top half of matrix).
    # In Flipped image, Top half corresponds to Original Bottom Half (Negative velocities).
    # But `relevant_velocity` is still 0 to 150 (Positive numbers).
    # So it finds a "Positive" envelope on the inverted negative data.
    # Then we negate the result. Correct.
    
    val = find_contiguous_max(image_bw, velocity)
    return -val

def spectrogram_tracing(vSP, SP, fs, ix_exclude_Otsu=None):
    # SP values to [0, 1]
    SP_norm = SP.copy()
    SP_norm[SP_norm <= 0] = 0
    max_val = np.max(SP_norm)
    if max_val > 0:
        SP_norm = SP_norm / max_val # mat2gray roughly scales min-max to 0-1. Here min is 0.
        
    rows, cols = SP_norm.shape
    if cols < 5 * fs or np.isnan(fs) or fs <= 1e-6:
        return np.zeros(cols), np.zeros(cols), np.zeros((rows, cols)), 0
        
    best_cbfv_qual = -float('inf')
    cbfv_env = np.zeros(cols)
    cbfv_sqi = np.zeros(cols)
    img_bw_best = np.zeros((rows, cols))
    multiplier_best = 0
    
    otsu_mlt = [1, 0.95, 0.9, 1.05, 1.1]
    
    # Indices
    # Assuming vSP is sorted.
    def get_idx(val):
        return np.argmin(np.abs(vSP - val))
        
    ix_neg130 = get_idx(-125)
    ix_pos130 = get_idx(125)
    ix_neg100 = get_idx(-100)
    ix_pos100 = get_idx(100)
    ix_neg20 = get_idx(-20)
    ix_pos20 = get_idx(20)
    ix_neg5 = get_idx(-5)
    ix_pos5 = get_idx(5)
    
    # Energy comparison
    # SPlower = SP(ix_neg100:ix_neg20, :); (Python slice exclusive)
    # Matlab indices are increasing? vSP usually -Max to +Max.
    # So -100 is lower index than -20.
    
    sp_lower = SP_norm[ix_neg100 : ix_neg20+1, :]
    sp_upper = SP_norm[ix_pos20 : ix_pos100+1, :]
    e_lower = np.sum(sp_lower)
    e_upper = np.sum(sp_upper)
    
    # Region for Otsu
    if e_lower > e_upper:
        sp_otsu = SP_norm[ix_neg130 : ix_neg5+1, :]
    else:
        sp_otsu = SP_norm[ix_pos5 : ix_pos130+1, :]
        
    # Loop
    for mlt in otsu_mlt:
        # Binarize
        # Filter small values from Otsu calculation?
        # SP_Otsu(SP_Otsu < 0.025) = []; in Matlab removes pixels.
        # skimage otsu takes array. Flatten it and remove small?
        
        flat_otsu = sp_otsu.flatten()
        flat_otsu = flat_otsu[flat_otsu >= 0.025]
        
        if len(flat_otsu) == 0:
            thresh_val = 0.5 # Fallback
        else:
            thresh_val = threshold_otsu(flat_otsu)
            
        thresh = mlt * thresh_val
        img_bw_tmp = SP_norm > thresh
        
        # Check white speckles
        if e_lower > e_upper:
             # topSeg = imgBW_Tmp(ix_neg130:ix_neg100, :);
             top_seg = img_bw_tmp[ix_neg130 : ix_neg100+1, :]
        else:
             top_seg = img_bw_tmp[ix_pos100 : ix_pos130+1, :]
             
        m_seg, n_seg = top_seg.shape
        if np.sum(top_seg) > (m_seg * n_seg / 3):
            thresh = 1.1 * thresh
            img_bw_tmp = SP_norm > thresh
            
        # Median filtering
        v_del = vSP[-1] - vSP[-2]
        if v_del == 0: v_del = 1 # Safety
        row_k = int(round(2.5 / v_del))
        col_k = int(round(0.05 * fs))
        
        # medfilt2d kernel size must be odd
        if row_k % 2 == 0: row_k += 1
        if col_k % 2 == 0: col_k += 1
        
        img_bw_tmp = signal.medfilt2d(img_bw_tmp.astype(float), kernel_size=[row_k, col_k])
        img_bw_tmp = img_bw_tmp > 0.5 # Binarize back
        
        # Retrieve CBFV
        direction = 1
        if e_lower > e_upper:
            img_bw_neg = np.flipud(img_bw_tmp)
            cbfv_tmp = find_contiguous_min(img_bw_neg, vSP)
            direction = -1
        else:
            cbfv_tmp = find_contiguous_max(img_bw_tmp, vSP)
            
        # SQA
        if len(cbfv_tmp) > 10 * fs:
            curr_qual, cbfv_sqi_tmp = sqa_cbfv(direction * cbfv_tmp, fs)
        else:
            curr_qual = 0
            cbfv_sqi_tmp = np.zeros(len(cbfv_tmp))
            
        if curr_qual > best_cbfv_qual:
            best_cbfv_qual = curr_qual
            cbfv_sqi = cbfv_sqi_tmp
            cbfv_env = cbfv_tmp
            img_bw_best = img_bw_tmp
            multiplier_best = mlt
            
    return cbfv_env, cbfv_sqi, img_bw_best, multiplier_best

# -------------------------------------------------------------------------
# MTCM Method Helpers
# -------------------------------------------------------------------------

def env_mtcm(vel, I, I_th, dur_th):
    # vel: 1D array
    # I: 1D array of intensities at this time step
    # I_th: threshold
    # dur_th: duration (samples)
    
    # Determine Above Threshold
    I_gt = (I > I_th).astype(int)
    
    env_vel = 0
    # Iterate backwards
    # for nn = length(vel):-1:(dur_th+1)
    # Matlab 1-based. length(vel) down to dur_th+1.
    # Python: len(vel)-1 down to dur_th.
    
    seg = np.ones(dur_th)
    
    # Indices mapping is tricky.
    # I_gt((nn-dur_th+1):nn) in Matlab is length dur_th.
    # If nn is index, range is [nn-dur_th+1, nn].
    # Python slice [nn-dur_th+1 : nn+1] ? No, python is 0 based.
    
    n = len(vel)
    for i in range(n - 1, dur_th - 1, -1):
        # Check segment of length dur_th ending at i
        # slice: i - dur_th + 1 to i + 1
        window = I_gt[i - dur_th + 1 : i + 1]
        if np.array_equal(window, seg):
            env_vel = vel[i]
            break
            
    return env_vel

def eval_env(I, vel, hld_max, ord_medfilt_pol, method, vel_noise_min, rat_noise_max, mult_th, dur_th):
    # I: spectrogram image (intensities) [Vel, Time]
    # vel: velocity axis
    
    rows, cols = I.shape
    env_vel_neg = np.zeros(cols)
    env_vel_pos = np.zeros(cols)
    pol = np.zeros(cols)
    
    vel_pos_mask = vel > 0
    vel_neg_mask = vel < 0
    vel_pos = vel[vel_pos_mask]
    vel_neg = vel[vel_neg_mask]
    
    # Noise Calcs
    # Mean noise intensity
    mask_noise_pos = vel > vel_noise_min
    mask_noise_neg = vel < -vel_noise_min
    
    # Matlab mean(I(mask, :)) ? No, mean of all pixels in that band?
    # Code: mean(I(vel>..., :)) implies taking all columns? 
    # It returns a row vector if I is matrix. 
    # But then I_noise_pos/neg comparison results in scalar?
    # "if ((I_noise_pos/I_noise_neg) > rat_noise_max)" implies scalars.
    # So it's mean over entire band (all time).
    
    i_noise_pos = np.mean(I[mask_noise_pos, :])
    i_noise_neg = np.mean(I[mask_noise_neg, :])
    
    if i_noise_neg == 0: i_noise_neg = 1e-9 # safe div
    
    if (i_noise_pos / i_noise_neg) > rat_noise_max:
        i_noise_avg = i_noise_neg
    elif (i_noise_neg / i_noise_pos) > rat_noise_max:
        i_noise_avg = i_noise_pos
    else:
        i_noise_avg = np.mean([i_noise_pos, i_noise_neg])
        
    i_th = i_noise_avg * mult_th
    
    cnt_hold_pos = 0
    cnt_hold_neg = 0
    
    for nn in range(cols):
        I_col = I[:, nn]
        I_pos = I_col[vel_pos_mask]
        I_neg = I_col[vel_neg_mask]
        
        pol[nn] = 1 if np.sum(I_pos) > np.sum(I_neg) else 0
        
        if method == 'MTCM':
            env_vel_pos[nn] = env_mtcm(vel_pos, I_pos, i_th, dur_th)
            # flipud vel_neg and I_neg
            env_vel_neg[nn] = env_mtcm(vel_neg[::-1], I_neg[::-1], i_th, dur_th)
            
        # Zero Order Hold
        if env_vel_pos[nn] == 0 and nn != 0 and cnt_hold_pos < hld_max:
            env_vel_pos[nn] = env_vel_pos[nn-1]
            cnt_hold_pos += 1
        else:
            cnt_hold_pos = 0
            
        if env_vel_neg[nn] == 0 and nn != 0 and cnt_hold_neg < hld_max:
            env_vel_neg[nn] = env_vel_neg[nn-1]
            cnt_hold_neg += 1
        else:
            cnt_hold_neg = 0
            
    # Median filter polarity
    if ord_medfilt_pol % 2 == 0: ord_medfilt_pol += 1
    pol_filt = signal.medfilt(pol, kernel_size=ord_medfilt_pol)
    
    # Dual Sided
    env_vel = env_vel_pos * pol_filt + env_vel_neg * (1 - pol_filt)
    
    return env_vel

def run_ultrasound_tracing_mtcm(v_spectrogram, SP):
    # Configuration
    hld_max = 25
    ord_medfilt_pol = 10 # make odd later
    method = 'MTCM'
    vel_noise_min = 1.8
    rat_noise_max = 2
    mult_th = 1.5
    dur_th = 4
    
    I_avg_filt = SP
    env_vel = eval_env(I_avg_filt, v_spectrogram, hld_max, ord_medfilt_pol, method, vel_noise_min, rat_noise_max, mult_th, dur_th)
    return env_vel

def ultrasound_tracing_mtcm(v_spectrogram, SP):
    # Filter Spectral Image
    proc_size_medimfilt = [7, 7]
    proc_size_gaussimfilt = [15, 15]
    sigma_gauss = 0.025 # This is for imgaussfilt.
    # imgaussfilt sigma is standard deviation. 0.025 pixels? That's tiny. 
    # Maybe the code meant something else or the image is scaled? 
    # If image is standard spectrogram, 0.025 is very small. 
    # However, I will stick to the value.
    
    sp_avg_filt = signal.medfilt2d(SP, kernel_size=proc_size_medimfilt)
    sp_avg_filt = ndimage.gaussian_filter(sp_avg_filt, sigma=sigma_gauss) # default trunc is 4.0
    
    cbfv_env_pos_mtcm = run_ultrasound_tracing_mtcm(v_spectrogram, sp_avg_filt)
    cbfv_env_neg_mtcm = cbfv_env_pos_mtcm - 1
    
    if np.sum(np.abs(cbfv_env_pos_mtcm)) > np.sum(np.abs(cbfv_env_neg_mtcm)):
        return cbfv_env_pos_mtcm
    else:
        return cbfv_env_neg_mtcm
