function [fSpectrogram, tSpectrogram, SP, fs_Spec] = generate_spectrogram(IQ, fs, t)

% Parameters for spectrum generation
nsamples = 128; % Window size for spectrogram.
fracoverlap = 0.75; % Extent of window overlap
NFFT = 1024; % Zero-padding
fwc = 100; % Wall filter cutoff freq [Hz]

% Compute Doppler spectrogam
[~, fSpectrogram, tSpectrogram, SP] = compute_spectrogram(IQ, fwc, fs, nsamples, fracoverlap, NFFT);  
fs_Spec = 1/median(diff(tSpectrogram));
tSpectrogram = tSpectrogram + t(1);
end

