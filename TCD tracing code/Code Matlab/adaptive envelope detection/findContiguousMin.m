function cbfv_Env = findContiguousMin(imageBW,velocity)

% Flip around imageBW and SP
cbfv_Env = findContiguousMax(imageBW,velocity);
cbfv_Env = -cbfv_Env;
