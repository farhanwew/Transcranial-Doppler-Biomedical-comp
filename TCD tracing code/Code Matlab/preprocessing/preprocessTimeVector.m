function t = preprocessTimeVector(t)

% Zero the timer and convert milliseconds to seconds
t = (t - t(1))/10^6;

% Sampling rate
diffT = diff(t);
Ts = median(diffT);

% Correct for overflow in time vector (detect negative jumps in time stamps)
overflowIndex = find(diffT < -Ts);
overflowIndex = [overflowIndex, length(t)];

for i = 1:length(overflowIndex)-1
    DeltaT = abs(t(overflowIndex(i)) - t(overflowIndex(i)+1)) + Ts;
    idx = overflowIndex(i)+1:overflowIndex(i+1);
    t(idx) = t(idx) + DeltaT;
end

end

