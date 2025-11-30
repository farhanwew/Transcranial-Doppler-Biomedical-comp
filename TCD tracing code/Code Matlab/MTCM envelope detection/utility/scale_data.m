%++++++++++> scale_data <++++++++++++++++++++++++++++++++++++++++++++++++++
function data_scl = scale_data(data,data_clip_min,data_clip_max,map_min,map_max,func_tran,fac_tran)
% Clip Input Data
data(data<data_clip_min) = data_clip_min;
data(data>data_clip_max) = data_clip_max;

% Apply Data Transformation
if (isempty(fac_tran))
    data_tran = func_tran(data);
    data_clip_min_tran = func_tran(data_clip_min);
    data_clip_max_tran = func_tran(data_clip_max);
else
    data_tran = func_tran(data,fac_tran);
    data_clip_min_tran = func_tran(data_clip_min,fac_tran);
    data_clip_max_tran = func_tran(data_clip_max,fac_tran);
end

% Scale Data To Mapping Limits
data_scl = (data_tran-data_clip_min_tran)*(map_max-map_min)/(data_clip_max_tran-data_clip_min_tran)+map_min;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++