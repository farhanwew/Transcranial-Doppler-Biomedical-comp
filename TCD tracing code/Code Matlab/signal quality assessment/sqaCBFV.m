function [goodSigPercent, sqiSignal] = sqaCBFV(cbfv, fs)
% Compute signal quality of CFBV signal using a decision tree checking the amplitude, beat duration and the fit 
% to a template learned from the data

% Detect beat onsets
beatOnsets = wabp_wrapper(cbfv, fs);
segDuration = length(cbfv)/(60*fs);

% If too few beat onsets found (< 10 bpm) or maximal flow velocity very high (hints
% to error mode where waveform tracks the noise, set the signal quality to zero.
if(length(beatOnsets)/segDuration < 10 || median(cbfv) > 120)
    goodSigPercent = 0;
    sqiSignal = zeros(1, length(cbfv));
else
    binaryLabelAmp = amplitudeBasedLabeling(cbfv, beatOnsets); 
    [binaryLabelPeriod, cbfvTemplate] = periodBasedLabeling(cbfv, beatOnsets, binaryLabelAmp, fs); 
    [binaryLabelModel, sqiSignal] = templateBasedLabeling(cbfv, beatOnsets, binaryLabelPeriod, cbfvTemplate, fs); 
    artifacts = find(binaryLabelModel);
    goodSigPercent = 1 - length(artifacts)/length(cbfv);
end
end

