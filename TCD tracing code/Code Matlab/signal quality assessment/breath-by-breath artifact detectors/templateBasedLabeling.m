function [isArtifact, sqiSignal] = templateBasedLabeling(cbfv, beatOnsets, artifactLabel, medianTemplate, fs)
% Template matching for signal quality assessment.
     
% Let template start at zero and scale to 1
lenCbfv = length(cbfv); 
isArtifact = ones(1, lenCbfv);   
sqiSignal = zeros(1, lenCbfv);
sqiSignal(1:beatOnsets(1)) = nan; % Signal quality assessment start after first detected beat till last
sqiSignal(beatOnsets(end):end) = nan;

% Take only part of medianTemplate, since the last part is flat (contains zeros)
medianTemplate(medianTemplate < 10) = [];

if(isempty(medianTemplate))
    return;
end
  
% Sanity check on template. If pulse amplitude too small we assume that the
% whole segment had more than 50 % outlier beats.
pulsatilityTemplate = max(medianTemplate) - min(medianTemplate);

if(pulsatilityTemplate < 25 || pulsatilityTemplate < 0.5*mean(medianTemplate))
    return;
end

%medianTemplate = medianTemplate - medianTemplate(1);
%medianTemplate = medianTemplate/max(medianTemplate);

% Go through every beat that was not labeled by amplitude labeling and assign label good/bad
for i = 1:length(beatOnsets) - 1
    pulseWindow = beatOnsets(i):beatOnsets(i+1)-1;   
    pulseWave = cbfv(pulseWindow)';
    amplitudeOrPeriodArtifact = sum(artifactLabel(pulseWindow(2:end-1)));

    if(~amplitudeOrPeriodArtifact) 
        [artifact_dueToMSE, sqi] = nrmsClassification(medianTemplate, pulseWave, fs);
        sqiSignal(pulseWindow) = sqi*ones(1,length(pulseWindow));

        if(~artifact_dueToMSE)    
            isArtifact(pulseWindow) = zeros(1,length(pulseWindow)); 
        end
    end
end
end

