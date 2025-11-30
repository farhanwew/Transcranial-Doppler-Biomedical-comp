function cbfvEnvMtcm = ultrasound_tracing_mtcm(vSpectrogram, SP)

  % Spectral image filter
  proc.dur_avg = 8; % Averaging Duration, Spectra
  proc.size_medimfilt = [7, 7]; % Median Image Filter Kernel Size [Velocity Axis, Time Axis], Pixels
  proc.size_gaussimfilt = [15, 15]; % Gaussian Image Filter Kernel Size [Velocity Axis, Time Axis], Pixels
  proc.sigma_gaussimfilt = 0.025; % Gaussian Image Filter Standard Deviation
  
  % Filter Spectral Image
  SP_avg_filt = medfilt2(SP, proc.size_medimfilt);
  SP_avg_filt = imgaussfilt(SP_avg_filt, proc.sigma_gaussimfilt, 'FilterSize', proc.size_gaussimfilt, 'Padding', 'replicate');
  cbfvEnvPosMtcm = run_ultrasound_tracing_mtcm(vSpectrogram, SP_avg_filt); 
  cbfv_Env_NegMtcm = cbfvEnvPosMtcm - 1;  % such that only positive gets displayed 

  if(sum(abs(cbfvEnvPosMtcm)) > sum(abs(cbfv_Env_NegMtcm)))
    cbfvEnvMtcm = cbfvEnvPosMtcm;
  else
    cbfvEnvMtcm = cbfv_Env_NegMtcm;  
  end 
end

