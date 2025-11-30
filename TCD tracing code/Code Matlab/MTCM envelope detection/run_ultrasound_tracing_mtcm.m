function [env_vel] = run_ultrasound_tracing_mtcm(v_spectrogram, SP)


% Spectral Envelope Generation
proc.hld_max         = 25;      % Zero-Order Hold Maximum, Samples
proc.ord_medfilt_pol = 10;      % Polarity Median Filter Order
proc.method          = 'MTCM';  % Envelope Computation Method ['MTCM' (Preferred)]
proc.vel_noise_min   = 1.8;     % Minimum Noise Velocity Magnitude, m/s
proc.rat_noise_max   = 2;       % Maximum Average Noise Intensity Ratio
proc.mult_th         = 1.5;    % Noise Multiple Threshold [8 for SNR = 5 to 13 dB, 11 for SNR = 13 to 20 dB; "Computer Evaluation of Doppler Spectral Envelope Area In Patients Having A Valvular Aortic Stenosis" - G. Cloutier]
proc.dur_th          = 4;       % Consecutive Supra-Threshold Duration, Samples

% Velocity Envelope
I_avg_filt = SP;
proc.vel   = v_spectrogram;
env_vel    = eval_env(I_avg_filt, proc.vel, proc.hld_max, proc.ord_medfilt_pol, ...
                      proc.method, proc.vel_noise_min, proc.rat_noise_max, ...
                      proc.mult_th, proc.dur_th);
end

