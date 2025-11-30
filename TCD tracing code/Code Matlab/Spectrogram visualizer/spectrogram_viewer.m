%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spectrogram visualization script for echo data from Philips CX-50
% ultrasound machine.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%x
clear all;
close all;
clc;
addpath(genpath('./'));

% Parameters
fc = 1.75e6; % Transmit center frequency (carrier freq.)
ss = 1540; % Assumed speed of sound [m/s]
angle = 0; % Doppler angle [deg]
sizeSPfilt = [3, 3]; % Kernel size for median filter

% Load echo and generate IQ signal
filepath = '../Healthy Subjects/';
filename = 'Healthy_Subjects_Recording_1.txt';
data = readtable(strcat(filepath, filename));
tEcho = data.t;
I = data.I;
Q = data.Q;

% Zero the timestamp and convert milliseconds to seconds
tEcho = (tEcho - tEcho(1))/10^6;
diffT = diff(tEcho);
TsEcho = median(diffT);

% Correct for potential overflow in time vector (detect negative jumps in time vector)
overFlowIndex = find(diffT < -TsEcho);
overFlowIndex = [overFlowIndex, length(tEcho)];

for i = 1:length(overFlowIndex) - 1
    DeltaT = abs(tEcho(overFlowIndex(i)) - tEcho(overFlowIndex(i)+1)) + TsEcho;
    win = overFlowIndex(i)+1:overFlowIndex(i+1);
    tEcho(win) = tEcho(win) + DeltaT;
end

% Resample IQ signal
TsEcho = 1.4400e-04; % T=1/fs such that velcity range till 152 cm/s
fsEcho = 1/TsEcho;
tUniform = tEcho(1):TsEcho:tEcho(end);
I = (interp1(tEcho, I, tUniform, 'pchip'))';
Q = (interp1(tEcho, Q, tUniform, 'pchip'))';
IQ = complex(I, Q);
tEcho = tUniform;

% Compute spectrogram
[freqs_SP, tSpectrogram, SP, fs_SP] = generate_spectrogram(IQ, fsEcho, tEcho);

% Convert frequency axis to velocity
vSpectrogram = 100*freqs_SP*ss/(2*fc*cosd(angle));

% Doppler spectrum processing
SP = log2(SP);
SP = SP/max(SP(:)); 
SP = medfilt2(SP, sizeSPfilt);

% Visualize spectrogram
imagesc(tSpectrogram, vSpectrogram, SP); hold on;
h1 = gca; 
set(h1, 'clim', [0.1, 0.95]);
axis xy; 
colormap(gray); 
axis on;
xlabel('Time [s]'); 
ylabel('CBFV [cm/s]');