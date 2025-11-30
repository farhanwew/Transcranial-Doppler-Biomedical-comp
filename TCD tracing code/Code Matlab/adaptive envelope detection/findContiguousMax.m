function cbfv_Env = findContiguousMax(imageBW, velocity)

[~, colLen] = size(imageBW);
prevVelEnv = 0;
num_missDet = 0;
emptyMax = 0;
cbfv_Env = zeros(1, colLen);

% Bounds for abs(CBFV)
DelV = velocity(2) - velocity(1);
[~, minEnv] = min(abs(velocity));  
velocity = velocity(minEnv:end);

% Iterate over each time step and determine maximal velocity
for col = 1:colLen
  rawBWImgCol = imageBW(minEnv:end, col);

  % Get 10 maximal frequency candidate values  
  idxMax = find(rawBWImgCol); 
  idxMax = sort(idxMax, 'descend');
  idxMax = idxMax(1:min(10, length(idxMax)));
  
  if(isempty(idxMax))
    emptyMax = emptyMax + 1; 
    cbfv_Env(col) = prevVelEnv; 
  else
    cadidateCbfvs = velocity(idxMax);
    emptyMax = 0;
  
    % Iterative over each candidate velocity and check constraints
    for i = 1:length(cadidateCbfvs)
      
      % 1.) The majority of pixels beneath the candidate is white
      if(idxMax(i) > round(12/DelV))
        onesMaj = mode(rawBWImgCol((idxMax(i) - round(12/DelV)):idxMax(i)));
      else
        onesMaj = 1;
      end
     
      % 2.) Current value not more than 30 cm/s away from previous 3 values
      if(col > 4)
        largeDev = (cadidateCbfvs(i) - mean(cbfv_Env(col-4:col-1)) > 30);
      else
        largeDev = 0;
      end
      
      % If candidate velocities rejected for too many timesteps,
      if(onesMaj && ~largeDev && i<=2)
        num_missDet = 0;
      else
        num_missDet = num_missDet + 1;
      end

      % Check conditions and find candidate value
      if((onesMaj && ~largeDev) || num_missDet > 20)
        cbfv_Env(col) = cadidateCbfvs(i);
        prevVelEnv = cadidateCbfvs(i);
        num_missDet = 0;
        break;
      elseif(i == length(cadidateCbfvs))
        cbfv_Env(col) = prevVelEnv;
      end
    end
  end
end
end




