function [artifact, sqi] = nrmsClassification(templatePulse, candidatePulse, fs)
% Assign class label {0,1} on a beat-by-beat basis according to the MSE 
% between the template pulse and the candidate pulse

% Options
showPlt = 0;

% Initial values
maxNrmse = 0.25;
lenTemplate = length(templatePulse);
lenCandidate = length(candidatePulse);
templatePulse = templatePulse - templatePulse(1);
templatePulse = templatePulse/max(templatePulse);
candidatePulse = candidatePulse - candidatePulse(1);
candidatePulse = candidatePulse/max(candidatePulse);
  
% If candidate shorter than template, fill up the candidate beat
if(lenTemplate > lenCandidate)
    candidatePulse = [candidatePulse, candidatePulse(end)*ones(1,length(templatePulse) - length(candidatePulse))];
end
    
% Align candidate and template pulse
[r, lags] = xcorr(templatePulse, candidatePulse, round(0.15*fs));
[~, ix]=max(r);
lagBest = lags(ix);

if(lagBest == 0)
    % Alignment already good
elseif(lagBest < 0)
    templatePulse = templatePulse(1:end+lagBest);
    candidatePulse = candidatePulse(-lagBest:end);  
else
    templatePulse = templatePulse(lagBest:end);
    candidatePulse = candidatePulse(1:end-lagBest);   
end

windowMSE = 1:round(0.75*length(templatePulse)); % just shift one of the two
errorSigFW = templatePulse(windowMSE) - candidatePulse(windowMSE);
nrmse = sqrt(min(1, sum(errorSigFW.^2)/sum(templatePulse(windowMSE).^2)));

% NMSE thresholding
if(nrmse < maxNrmse)    
    artifact = 0;
else
    artifact = 1;
end
sqi = 1 - nrmse;

if(showPlt)
    figure;
    plot(templatePulse, 'k'); hold on;
    plot(templatePulse(windowMSE), '--g');
    plot(candidatePulse, 'r');
    title("NMSE = " + nrmse);
    pause;
    %keyboard;
    close;
end
end

