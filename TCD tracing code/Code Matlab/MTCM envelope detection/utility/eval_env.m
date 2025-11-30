%++++++++++> eval_env <++++++++++++++++++++++++++++++++++++++++++++++++++++
function [ env_vel ] = eval_env(I,vel,hld_max,ord_medfilt_pol,method,vel_noise_min,rat_noise_max,mult_th,dur_th)
% Instantiate Arrays
env_vel_neg = zeros(1,size(I,2));
env_vel_pos = zeros(1,size(I,2));
pol = zeros(1,size(I,2));

% Single-Sided Velocity Arrays, m/s
vel_pos = vel(vel>0);
vel_neg = vel(vel<0);

% Mean Noise Intensity
I_noise_pos = mean(I(vel>vel_noise_min,:));
I_noise_neg = mean(I(vel<(-vel_noise_min),:));
if ((I_noise_pos/I_noise_neg) > rat_noise_max)
    I_noise_avg = I_noise_neg;
elseif ((I_noise_neg/I_noise_pos) > rat_noise_max)
    I_noise_avg = I_noise_pos;
else
    I_noise_avg = mean([I_noise_pos, I_noise_neg]);
end

% Intensity Threshold
I_th = I_noise_avg*mult_th;

% Compute Spectral Envelope
cnt_hold_pos = 0;
cnt_hold_neg = 0;
for nn = 1:size(I,2)
    % Single-Sided Intensities
    I_pos = I(vel>0,nn);
    I_neg = I(vel<0,nn);

    % Determine Dominant Intensity Polarity
    pol(nn) = (sum(I_pos)>sum(I_neg));
    
    % Single-Sided Velocity Envelope, m/s
    if (strcmp(method,'MTCM'))
        env_vel_pos(nn) = env_mtcm(vel_pos,I_pos,I_th,dur_th);
        env_vel_neg(nn) = env_mtcm(flipud(vel_neg),flipud(I_neg),I_th,dur_th);
    else
        error('Envelope computation method undefined.');
    end
        
    % Zero-Order Hold
    if ((env_vel_pos(nn)==0) && (nn~=1) && (cnt_hold_pos<hld_max))
        env_vel_pos(nn) = env_vel_pos(nn-1);
        cnt_hold_pos = cnt_hold_pos+1;
    else
        cnt_hold_pos = 0;
    end
    if ((env_vel_neg(nn)==0) && (nn~=1) && (cnt_hold_neg<hld_max))
        env_vel_neg(nn) = env_vel_neg(nn-1);
        cnt_hold_neg = cnt_hold_neg+1;
    else
        cnt_hold_neg = 0;
    end
end

% Median Filter Dominant Intensity Polarity
pol_filt = medfilt1(pol,ord_medfilt_pol);

% Dual-Sided Velocity Envelope, m/s
env_vel = env_vel_pos.*pol_filt+env_vel_neg.*(~pol_filt);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++