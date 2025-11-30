%++++++++++> dtft <++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [I, data_rem] = dtft(data,win,len_fft,step_win,scl_win)
% Spectral Evaluations
K = floor((length(data)-length(win))/step_win+1);

% Evaluated Segment Length, Samples
N = length(win)+step_win*(K-1);

% Extract Data Segment
if (N == length(data))
    data_rem = [];
else
    data_rem = data((K*step_win+1):end);
    data = data(1:N);
end

% Preallocate Intensity Array
I = zeros(len_fft,K);

% Spectra Evaluation
for nn = 1:K
    I(:,nn) = fftshift(abs(fft(data((nn-1)*step_win+(1:length(win))).*win,len_fft)/(length(win)*scl_win)).^2);
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++