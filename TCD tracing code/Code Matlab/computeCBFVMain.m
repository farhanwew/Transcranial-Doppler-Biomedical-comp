%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Launch script for computing cerebral blood flow velocity (CBFV) waveforms
% from transcranial Doppler echos.
%
% Transcranial Doppler data recorded with the Philips CX50 can be found on
% the IEEE DataPort under: 
% F. Wadehn, T. Heldt, "Transcranial Doppler Ultrasound Database 
% (Philips CX50 device)", IEEE Dataport, 2020. [Online]. 
% Available: http://dx.doi.org/10.21227/44mg-2965. 
%
% The theoretical background is described in the paper
% F. Wadehn and T. Heldt, "Adaptive Maximal Blood Flow Velocity Estimation 
% from Transcranial Doppler Echos," in IEEE Journal of Translational
% Engineering in Health and Medicine, doi: 10.1109/JTEHM.2020.3011562.
%
% Federico Wadehn <wadehn.federico@gmail.com>
% ETH Zurich, Switzerland
% Created: August 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('./'));
clear all;
close all;
clc;

% Options
runAdaptiveEnvDetection = 1; % Our proposed approach
runMtcmEnvDetection = 1; % Maximum threshold crossing method

% Parameters
fc = 1.75e6; % Transmit center frequency (carrier freq.)
ss = 1540; % Assumed speed of sound [m/s]
angle = 0; % Doppler angle [deg]
sizeSPfilt = [3,3]; % Kernel size for median filter
otsuMltBest = []; % Otsu multiplier

% Initializations
cbfvMtcm = [];
cbfvAdaptive = [];
cbfvSqi = [];
tSpectrogram = [0];
SpBnW = [];
SP = [];

% Path to recordings
filepath = '../Healthy Subjects/';
filename = 'Healthy_Subjects_Recording_1.txt';
rawEchoData = readtable(strcat(filepath, filename));

% Compute envelope for only a segment of the data if recording is too long
% rawEchoData = rawEchoData(1:round(height(rawEchoData)/10), :); 

% Preprocess time vector
t = preprocessTimeVector(rawEchoData.t);

% Segment echo data into x-min segments
tcdEchoData = segmentEchoSignal(rawEchoData.I, rawEchoData.Q, t);

% Iterate over all segments, compute spectrogram and trace envelope
for i = 1:numel(tcdEchoData.t)
    
    % Generate spectrogram from IQ echo signal
    iqSignal = tcdEchoData.IQ{i};
    fsEcho = tcdEchoData.fs(i);
    tEcho = tcdEchoData.t{i};
    [freqs_SP, t_SP, SPseg, fs_SP] = generateSpectrogram(iqSignal, fsEcho, tEcho);
    tSpectrogram = [tSpectrogram, t_SP];
    ix_exclude_Otsu = round(tcdEchoData.ix_low_freq{i}*(fs_SP/fsEcho));
    
    % Convert frequency axis to velocity
    vSpectrogram = 100*freqs_SP*ss/(2*fc*cosd(angle));

    % Doppler spectrum processing
    SPseg = log2(SPseg);
    SPseg = SPseg/max(SPseg(:)); 
    SPseg = medfilt2(SPseg, sizeSPfilt);
    
    % Debug
    %imshow(SPseg);
    %axis xy;
    %keyboard;

    % Compute ultrasound spectrogram envelope
    if(runAdaptiveEnvDetection)
        [cbfvCurr, cbfvSqiCurr, SpCurrBW, multiplierBest] = spectrogramTracing(vSpectrogram, SPseg, fs_SP, ix_exclude_Otsu);
        cbfvCurr = postprocessCBFV(cbfvCurr, fs_SP);
       
        % Concatenate window-by-window CBFV estimates
        cbfvAdaptive = [cbfvAdaptive, cbfvCurr]; 
        cbfvSqi = [cbfvSqi, cbfvSqiCurr];
        SpBnW = [SpBnW, SpCurrBW];
        multiplierBest = multiplierBest*ones(1, length(cbfvCurr));
        otsuMltBest = [otsuMltBest, multiplierBest];
        SP = [SP, SPseg];
    end

    if(runMtcmEnvDetection) 
        cbfvCurr = ultrasound_tracing_mtcm(vSpectrogram, SPseg);
        cbfvCurr = postprocessCBFV(cbfvCurr, fs_SP);
        cbfvMtcm = [cbfvMtcm, cbfvCurr];
    end 
end

% Add blanks (nans) in spectrogram and envelope, where time vector has jumps
tSpectrogram = tSpectrogram(2:end);
ix_t_jump = find(diff(tSpectrogram) > 0.25);
Ts = median(diff(tSpectrogram));

for i = 1:length(ix_t_jump)
    ix_pre_jump = ix_t_jump(i);
    t_tmp = tSpectrogram(ix_pre_jump) + Ts : Ts : tSpectrogram(ix_pre_jump+1) - Ts;
    L = length(t_tmp);
    tSpectrogram = [tSpectrogram(1 : ix_pre_jump), t_tmp, tSpectrogram(ix_pre_jump + 1 : end)];
    cbfvAdaptive = [cbfvAdaptive(1 : ix_pre_jump), nan(1, L), cbfvAdaptive(ix_pre_jump + 1 : end)];
    cbfvMtcm = [cbfvMtcm(1 : ix_pre_jump), nan(1, L), cbfvMtcm(ix_pre_jump + 1 : end)];
    otsuMltBest = [otsuMltBest(1 : ix_pre_jump), nan(1, L), otsuMltBest(ix_pre_jump + 1 : end)];
    SP = [SP(:, 1 : ix_pre_jump), nan(length(vSpectrogram), L), SP(:, ix_pre_jump + 1 : end)];
    SpBnW = [SpBnW(:, 1 : ix_pre_jump), nan(length(vSpectrogram), L), SpBnW(:, ix_pre_jump + 1 : end)];
    cbfvSqi = [cbfvSqi(1 : ix_pre_jump), nan(1, L), cbfvSqi(ix_pre_jump + 1 : end)];
    ix_t_jump = ix_t_jump + L;
end

%% Plot spectrogram and envelope
figure;

% Show spectrogram (SP) with traced CBFV
ax(1) = subplot(4,1,1);
imagesc(tSpectrogram, vSpectrogram, SP); hold on;
plot(tSpectrogram, cbfvAdaptive, 'red', 'LineWidth', 2);
plot(tSpectrogram, cbfvMtcm, 'blue', 'LineWidth', 2); 
set(ax(1), 'clim', [0.1, 0.95]);
axis xy; 
colormap(gray); 
axis on;
xlabel('Time [s]'); 
ylabel('CBFV [cm/s]');
legend('Adaptive', 'MTCM');

% Show B&W thresholded spectrogram
ax(2) = subplot(4,1,2);
imagesc(tSpectrogram, vSpectrogram, SpBnW); 
hold on;
set(ax(2), 'clim', [0.1, 0.95]);
axis xy; 
colormap(gray); 
axis on;
xlabel('Time [s]'); 
ylabel('CBFV [cm/s]');

% Show cerebral blood flow velocity (CBFV)
ax(3) = subplot(4,1,3);  
plot(tSpectrogram, cbfvAdaptive, 'red', 'LineWidth', 2);
hold on;
xlabel('Time [s]');
ylabel('CBFV [cm/s]');
legend('Envelope');

% Signal quality index (Sqi)
ax(4) = subplot(4,1,4); 
plot(tSpectrogram, 100*cbfvSqi, 'red', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('SQI [%]');
linkaxes(ax, 'x');
xlim([0, tSpectrogram(end)]);