function [cbfvEnv, cbfvSQI, imgBW, multiplierBest] = spectrogramTracing(vSP, SP, fs, ix_exclude_Otsu)

% Options
show_plt = 0;

% Sanity checks (if spectrogram segment less than 5 seconds do not compute
% envelope)
[r, c] = size(SP);
if(c < 5*fs || isnan(fs))
   cbfvEnv = zeros(1, c);
   cbfvSQI = zeros(1, c);
   imgBW = zeros(r, c);
   multiplierBest = 0;
   return;
end

% Convert SP values to [0, 1] range
SP(SP <= 0) = 0;
SP = mat2gray(SP);
  
% Parameters and initial values
bestCbfvQual = -Inf;
currCbfvQual = 0;
cbfvSQItmp = zeros(1,length(SP(1,:)));
otsuMlt = [1, 0.95, 0.9, 1.05, 1.1];
numOtsuMultipliers = length(otsuMlt);
     
% Determine if positive or negative flow has more energy
[~, ix_neg130] = min(abs(vSP + 125));
[~, ix_pos130] = min(abs(vSP - 125));
[~, ix_neg100] = min(abs(vSP + 100));
[~, ix_pos100] = min(abs(vSP - 100));
[~, ix_neg20] = min(abs(vSP + 20));
[~, ix_pos20] = min(abs(vSP - 20));
[~, ix_neg5] = min(abs(vSP + 5));
[~, ix_pos5] = min(abs(vSP - 5));
SPlower = SP(ix_neg100:ix_neg20, :);
SPupper = SP(ix_pos20:ix_pos100, :);
Elower = sum(sum(SPlower));
Eupper = sum(sum(SPupper));
  
%% Apply hard-thresholding to get binary image
       
% Find threshold via Otsu method
if(Elower > Eupper)
    SP_Otsu = SP(ix_neg130:ix_neg5, :); % Exclude 0-5 cm/s due to Wall filter
else
    SP_Otsu = SP(ix_pos5:ix_pos130, :);
end

% if(~isempty(ix_exclude_Otsu))
%     ix_exclude_Otsu = unique(ix_exclude_Otsu);
%     ix_exclude_Otsu(ix_exclude_Otsu==0) = [];
%     ix_exclude_Otsu(ix_exclude_Otsu > length(SP_Otsu(1,:))) = []; 
%     SP_Otsu(:, ix_exclude_Otsu) = [];
% end
    
% Iterate through all segmentation thresholds and compute CBFV
for thresh_iter = 1:numOtsuMultipliers

    % Binarize image
    SP_Otsu(SP_Otsu < 0.025) = [];
    thresh = otsuMlt(thresh_iter)*graythresh(SP_Otsu);
    imgBW_Tmp = imbinarize(SP, thresh);
    
    % Sanity check if too many white speckles
    if(Elower > Eupper)
        topSeg = imgBW_Tmp(ix_neg130:ix_neg100, :);
    else
        topSeg = imgBW_Tmp(ix_pos100:ix_pos130, :);
    end
    
    % If too much white-space adapt grayscale threshold
    [m, n] = size(topSeg);
    
    if(sum(topSeg(:)) > m*n/3)
        thresh = 1.1*thresh;
    end

    % Median filtering to reduce remaining speckles 
    vDel = vSP(end) - vSP(end-1);
    row = round(2.5/vDel); % Average speckle height xx cm/s
    col = round(0.05*fs);  % Speckle duration xx s
    imgBW_Tmp = medfilt2(imgBW_Tmp, [row, col]);
    
    % Retrieve CBFV envelope of either top or bottom part
    mlt = 1;
      
    if(Elower > Eupper)
        imgBWNeg = flip(imgBW_Tmp);
        cbfvTmp = findContiguousMin(imgBWNeg, vSP);
        mlt = -1;
    else
        cbfvTmp = findContiguousMax(imgBW_Tmp, vSP);  
    end
      
    % Determine SQA of CBFV
    if(length(cbfvTmp) > 10*fs)
        [currCbfvQual, cbfvSQItmp] = sqaCBFV(mlt*cbfvTmp', fs);
    end
        
    % Keep CBFV with the best signal quality
    if(currCbfvQual > bestCbfvQual)
        bestCbfvQual = currCbfvQual;
        cbfvSQI = cbfvSQItmp;
        cbfvEnv = cbfvTmp;
        imgBW = imgBW_Tmp;
        multiplierBest = otsuMlt(thresh_iter);
    end
    
    % Visualization
    if(show_plt)
        figure; subplot(3,1,1);
        imagesc(1:length(SP(1,:)), vSP, SP); hold on;
        plot(1:length(SP(1,:)), cbfvEnv, 'r');
        h2 = gca; 
        set(h2, 'clim', [0.1 0.95]);
        axis xy; 
        colormap(gray);
    
        subplot(3,1,2);
        imagesc(1:length(SP(1,:)), vSP, imgBW_Tmp);
        h2 = gca; 
        set(h2, 'clim', [0.1 0.95]);
        axis xy; 
        colormap(gray);
    
        subplot(3,1,3);
        histogram(SP_Otsu); hold on;
        xline(graythresh(SP_Otsu), 'r', 'LineWidth', 2);
        xline(thresh, '--k', 'LineWidth', 2);
        title(thresh_iter);
        keyboard;
        close;
    end
end

end

