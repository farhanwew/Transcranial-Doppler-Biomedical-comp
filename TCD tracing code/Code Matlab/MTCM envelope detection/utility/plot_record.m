
data_baseband = 5E-5*rand(1E5,1);

%==========> Parameters <==================================================
%----------> Hardware Parameters <-----------------------------------------
s.param.f0 = 2E6;                           % Excitation Frequency, Hz
s.param.c = 1540;                           % Acoustic Phase Velocity, m/s [Tissue]
s.param.rate_pr = 10E3;                     % Pulse Repetition Rate, Hz

%--------------------------------------------------------------------------
%----------> Post-Processing Parameters <----------------------------------
% Highpass Filter
proc.hpf.ripple = 0.05;             % Passband Ripple, dB
proc.hpf.rej = 100;                 % Stopband Rejection, dB
proc.hpf.vel_clut_max = 4/100;      % Maximum Clutter Velocity, m/s
proc.hpf.vel_scat_min = 6/100;      % Minimum Scatterer Velocity, m/s
proc.hpf.len_offset = 0;            % Data Offset Length, Samples

% Spectral Generation
proc.len_win = 128;                 % Window Length, Samples
proc.step_win = 6;                  % Adjacent Window Step Size, Samples
proc.len_fft = 512;                 % FFT Length, Samples

% Spectral Image Filter
proc.dur_avg = 8;                   % Averaging Duration, Spectra
proc.size_medimfilt = [7 7];        % Median Image Filter Kernel Size [Velocity Axis, Time Axis], Pixels
proc.size_gaussimfilt = [15 15];    % Gaussian Image Filter Kernel Size [Velocity Axis, Time Axis], Pixels
proc.sigma_gaussimfilt = 0.025;     % Gaussian Image Filter Standard Deviation

% Spectral Envelope Generation
proc.hld_max = 25;                  % Zero-Order Hold Maximum, Samples
proc.ord_medfilt_pol = 10;          % Polarity Median Filter Order
proc.method = 'MTCM';               % Envelope Computation Method ['MTCM' (Preferred)]
proc.vel_noise_min = 1.8;           % Minimum Noise Velocity Magnitude, m/s
proc.rat_noise_max = 2;             % Maximum Average Noise Intensity Ratio
proc.mult_th = 8;                   % Noise Multiple Threshold [8 for SNR = 5 to 13 dB, 11 for SNR = 13 to 20 dB; "Computer Evaluation of Doppler Spectral Envelope Area In Patients Having A Valvular Aortic Stenosis" - G. Cloutier]
proc.dur_th = 4;                    % Consecutive Supra-Threshold Duration, Samples

% Spectral Envelope Filter
proc.ord_medfilt_env = 5;           % Envelope Median Filter Order
proc.ord_lpf_env = 256;             % Envelope Lowpass Filter Order
proc.freq_lpf_env = 25;             % Envelope Lowpass Filter Cutoff Frequency, Hz

% Image Display
proc.axis_v = [-150 180]/1E2;       % Velocity Axis Limits, m/s
proc.I_min = 3E-15;                 % Intensity Minimum Threshold
proc.I_max = 1E-7;                  % Intensity Maximum Threshold
proc.I_cm_min = 1;                  % Intensity Color Map Minimum
proc.I_cm_max = 64;                 % Intensity Color Map Maximum
proc.I_cm_func_tran = @nthroot;     % Intensity Color Map Transformation Function
proc.I_cm_fac_tran = 4;             % Intensity Color Map Transformation Factor

%--------------------------------------------------------------------------
%----------> Dependent Parameters <----------------------------------------
% Highpass Filter
proc.hpf.freq_clut_max = 2*proc.hpf.vel_clut_max*s.param.f0/s.param.c;
proc.hpf.freq_scat_min = 2*proc.hpf.vel_scat_min*s.param.f0/s.param.c;
[ord,Ws] = cheb2ord(2*proc.hpf.freq_scat_min/s.param.rate_pr,2*proc.hpf.freq_clut_max/s.param.rate_pr,proc.hpf.ripple,proc.hpf.rej);
[z,p,k] = cheby2(ord,proc.hpf.rej,Ws,'high');
[sos,g] = zp2sos(z,p,k);
proc.hpf.filt = dfilt.df2sos(sos,g);
proc.hpf.filt.persistentmemory = true;
proc.hpf.filt.state = 0;

% Spectral Envelope
proc.win = hann(proc.len_win);
proc.scl_win = sum(proc.win)/proc.len_win;
proc.rate_sample = s.param.rate_pr/proc.step_win/proc.dur_avg;
proc.freq = (s.param.rate_pr/proc.len_fft)*((-proc.len_fft/2):(proc.len_fft/2-1));
proc.vel = proc.freq*s.param.c/2/s.param.f0;

%--------------------------------------------------------------------------
%==========================================================================
%==========> Plot Data <===================================================
%----------> Spectral Envelope Data <--------------------------------------
% Filter Data
data_baseband_hpf = filter(proc.hpf.filt,data_baseband);    

% Data Spectra
[I, ~] = dtft(data_baseband_hpf,proc.win,proc.len_fft,proc.step_win,proc.scl_win);

% Spectral Averaging
I_avg = zeros(size(I,1),floor(size(I,2)/proc.dur_avg));
for nn = 1:size(I_avg,2)
    I_avg(:,nn) = mean(I(:,(nn-1)*proc.dur_avg+(1:proc.dur_avg)),2);
end

% Filter Spectral Image
I_avg_filt = medfilt2(I_avg,proc.size_medimfilt);
I_avg_filt = imgaussfilt(I_avg_filt,proc.sigma_gaussimfilt,'FilterSize',proc.size_gaussimfilt,'Padding','replicate');

% Velocity Envelope, m/s
env_vel = eval_env(I_avg_filt,proc.vel,proc.hld_max,proc.ord_medfilt_pol,proc.method,proc.vel_noise_min,proc.rat_noise_max,proc.mult_th,proc.dur_th);

% Filter Velocity Envelope
env_vel_filt = medfilt1(env_vel,proc.ord_medfilt_env);
env_vel_filt = filtfilt(fir1(proc.ord_lpf_env,2*proc.freq_lpf_env/proc.rate_sample),1,env_vel_filt);

% Time, s
env_t = (0:(length(env_vel_filt)-1))/proc.rate_sample;

% Display Spectrogram
figure();
hold on;
colormap(hot(proc.I_cm_max));
image(env_t,proc.vel*1E2,scale_data(I_avg_filt,proc.I_min,proc.I_max,proc.I_cm_min,proc.I_cm_max,proc.I_cm_func_tran,proc.I_cm_fac_tran));
set(gca,'XLim',[env_t(1) env_t(end)]);
set(gca,'YDir','Normal','YLim',1E2*[proc.vel(1) proc.vel(end)]);
colorbar('Location','EastOutside','YTick',[]);
ylabel(findall(gcf,'Type','colorbar'),'Normalized Spectral Intensity','FontSize',14);
plot(env_t,env_vel_filt*100,'-b','LineWidth',2.5);
xlabel('Time, s','FontSize',14);
ylabel('Flow Velocity, cm/s','FontSize',14);
set(gca,'FontSize',14);
hold off;
%--------------------------------------------------------------------------
%==========================================================================