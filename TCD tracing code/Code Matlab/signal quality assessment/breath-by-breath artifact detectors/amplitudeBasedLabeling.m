function isArtifact = amplitudeBasedLabeling(cbfv, beatOnsets)
% Amplitude-based physiological sanity checks for CBFV waveform
% 0: no artifac 
% 1: artifact

lenRec = length(cbfv);
isArtifact = zeros(1, lenRec);
minPulsatility = 20;
minAmplitude = 30;

% If no beats, return
if(isempty(beatOnsets))
  isArtifact = ones(1, lenRec);
  return;
end

% Go through each beat and assess its quality
for i = 1:length(beatOnsets)-1
  window = beatOnsets(i):beatOnsets(i+1) - 1;
  win_len = length(window);
  cbfv_wavelet = cbfv(window);
  min_cbfv = min(cbfv_wavelet);
  max_cbfv = max(cbfv_wavelet);
  
  % Conditions on min and max CBFV
  if(max_cbfv < minAmplitude)
    isArtifact(window) = ones(1, win_len);
    continue;
  end   
  
  % CBFV wavelets needs to have a minimal amount of pulsatility
  pulsatility = max_cbfv - min_cbfv;
  
  if(pulsatility < minPulsatility && max_cbfv > 45)
    isArtifact(window) = ones(1, win_len);
    continue;
  end
  
 % Pulsatility should be large compared to mean flow 
 if(pulsatility < 0.5*mean(cbfv_wavelet))
    isArtifact(window) = ones(1, win_len);
 end
  
  
end

end

