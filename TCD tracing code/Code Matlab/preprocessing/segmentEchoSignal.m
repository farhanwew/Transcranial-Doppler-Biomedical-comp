function tcd_echo = segmentEchoSignal(I, Q, t)
% SEGMENT_IQ_SIGNAL Segment received echo signal (more precisely, its
% demodulated IQ component) into 1 min segments.

% Initializations
t_cell = {};
IQ_cell = {};
fs_array = [];
ix_low_freq_cell = {};

% Independent parameters
windowLength = 60;

% Segment IQ data such that jumps in the time vector (e.g., above 0.25s) go into
% separate windows. Such jumps can occur when the insonation was paused and
% settings were changed.
diffT = diff(t);
ix_t_jumps = [0; find(diffT > 0.5); length(t)];
ix_segments = [];

% Get segments
for i = 1:length(ix_t_jumps) - 1
    win = ix_t_jumps(i) + 1 : ix_t_jumps(i+1);
    ix_ta = win(1);
    win_end_reached = 0;

    % Segment the window (if long enough) into segments of 60s
    while(true)
        if(t(ix_ta) + windowLength + windowLength - 5 > t(win(end)))
            tb = t(win(end));
            win_end_reached = 1;
        else
            tb = t(ix_ta) + windowLength;
        end
        [~, ix_tb] = min(abs(t - tb));
        
        ix_segments = [ ix_segments;
                       [ix_ta, ix_tb]
                       ];
        if(win_end_reached)
            break;
        else
            ix_ta = ix_tb + 1;
        end
    end
end

% Populate t and IQ cell arrays with signal segments
for i = 1:length(ix_segments(:,1))
    win = ix_segments(i,1) : ix_segments(i,2);
    [t_tmp, ix_unique, ~] = unique(t(win));
    Itmp = I(win);
    Itmp = Itmp(ix_unique);
    Qtmp = Q(win);
    Qtmp = Qtmp(ix_unique);

    % Find lower freqs in t_tmp
    ix_low_freq_cell{i} = find(diff(t_tmp) > 1.1*median(diff(t_tmp)));
    
    % Resample IQ signal
    Ts = 1.4400e-04; % T=1/fs such that velcity range till 152 cm/s
    fs = 1/Ts;
    tUniform = t_tmp(1):Ts:t_tmp(end);
    Itmp = (interp1(t_tmp, Itmp, tUniform, 'pchip'))';
    Qtmp = (interp1(t_tmp, Qtmp, tUniform, 'pchip'))';
    IQ = complex(Itmp, Qtmp);
    t_cell{i} = tUniform;
    IQ_cell{i} = IQ;
    fs_array(i) = fs;  
end

tcd_echo.t = t_cell;
tcd_echo.IQ = IQ_cell;
tcd_echo.fs = fs_array;
tcd_echo.ix_low_freq = ix_low_freq_cell;


