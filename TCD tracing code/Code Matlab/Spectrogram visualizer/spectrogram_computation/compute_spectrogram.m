function [S, fSpectrogram, tSpectrogramTmp, SP] = compute_spectrogram(IQ,fwc,fs,nsamples,fracoverlap,NFFT)

% Filter IQ data.
[b,a]     = butter(2, fwc / (fs/2),'high'); % Cutoff frequency / Nyquist
IQ        = filtfilt(b,a,IQ);
IQ_length = length(IQ);
IQ        = IQ(1:floor(IQ_length/nsamples)*nsamples,1);
  
% Generate spectrogram consisting of sliding power spectral densities from complex IQ.
[S, fSpectrogram, tSpectrogramTmp, SP] = spectrogram(IQ, hann(nsamples), fracoverlap*nsamples, NFFT, fs, 'centered');

end

