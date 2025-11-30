function [binaryLabel, medianTemplate] = periodBasedLabeling(cbfv, beatOnsets, binaryLabelAmp, fs)
% Signal quality assessment using relative and absolute beat durations
 
% Decision tree parameters
minDuration = 0.25; % 0.25s ~ 240 bpm
maxDuration = 2; % 2s ~ 30 bpm
diffOnsets = diff(beatOnsets);
medianPeriod = median(diffOnsets); 
minPeriod = 0.3*medianPeriod;
maxPeriod = 3*medianPeriod; 

% Initializations
pulseCnt = 0;
numBeats = length(beatOnsets);
pulse_collection = zeros(numBeats, max(diffOnsets));
pulse_length_vec = zeros(1, numBeats);
binaryLabel = zeros(length(cbfv), 1); 

for i = 1:length(beatOnsets) - 1
    pulseWindow = beatOnsets(i):beatOnsets(i+1);   
    pulseWave = cbfv(pulseWindow);
    ampArtifact = binaryLabelAmp(pulseWindow(1)); 

    % Pulse length deviates too much or AmpArtifact
    pulseDuration = diffOnsets(i)/fs;
    abs_duration_bad = pulseDuration < minDuration || pulseDuration > maxDuration;
    rel_duration_bad = diffOnsets(i) < minPeriod || diffOnsets(i) > maxPeriod;
    
    if(abs_duration_bad || rel_duration_bad || ampArtifact ~=0) 
        binaryLabel(pulseWindow) = 1; 
    else
        pulseCnt = pulseCnt + 1; 
        pulse_length = length(pulseWave); 
        pulse_length_vec(pulseCnt) = pulse_length;            
        pulse_collection(pulseCnt, 1:pulse_length) = pulseWave; 
    end 
end
medianTemplate = median(pulse_collection(1:pulseCnt, :)); 
end

