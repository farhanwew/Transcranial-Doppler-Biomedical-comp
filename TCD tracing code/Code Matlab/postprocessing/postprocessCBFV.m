function cbfv = postprocessCBFV(cbfv, fs)
%PREPROCESSCBFV Run lowpass filters.

if(isnan(fs))
    return;
end

fc = 50; 
fNyquist = fs/2;
order = 2;

% LPF
[b, a] = butter(order, fc/fNyquist);
cbfv = filtfilt(b, a, cbfv);
n = round(0.05*fs);
cbfv = medfilt1(cbfv, n);
end

