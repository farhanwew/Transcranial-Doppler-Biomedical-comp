%++++++++++> env_mtcm <++++++++++++++++++++++++++++++++++++++++++++++++++++
function env_vel = env_mtcm(vel,I,I_th,dur_th)
    % Determine Above Threshold Intensities
    I_gt = zeros(length(vel),1);
    I_gt(I>I_th) = 1;
    
    % Compute Envelope
    env_vel = 0;
    seg = ones(dur_th,1);
    for nn = length(vel):-1:(dur_th+1)
        if (I_gt((nn-dur_th+1):nn) == seg)
            env_vel = vel(nn);
            break;
        end
    end
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++