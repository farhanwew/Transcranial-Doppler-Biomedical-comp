function beatOnsets = wabp_wrapper(cbfv, fs)
%WABP_WRAPPER Processes the CBFV waveform such that the beats can be
%detected with the wabp function that was developed for arterial blood pressure
%waveforms. 

beatOnsets = [];

% Resample to 125 Hz, which is the sampling rate the wabp algo expects
fsNew = 125;
cbfvResampled = resample(cbfv, (0:numel(cbfv)-1)/fs, 125);
  
% Find median peak height
cbfv_pks = findpeaks(cbfvResampled, 'MinPeakHeight', 20, 'MinPeakDistance', 0.5*fsNew);

if(isempty(cbfv_pks))
    return;
end

% Find beat onsets
beatOnsets = wabp(140*cbfvResampled/median(cbfv_pks));

% Transform back to original sampling rate
beatOnsets = round(beatOnsets*(fs/fsNew));

% Small realignment of onset
winSize = round(0.05*fs);
lenSignal = length(cbfv);

for i=1:length(beatOnsets)
    lb = max(1, beatOnsets(i) - winSize);
    rb = min(lenSignal, beatOnsets(i) + winSize);
    [~, ixCorrected] = min(cbfv(lb:rb));
    beatOnsets(i) = lb + ixCorrected - 1;  
end


% Visualize detected onsets
%figure; plot(cbfv); hold on; plot(beatOnsets, cbfv(beatOnsets), 'rx');
%keyboard;

end

