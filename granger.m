%计算格兰杰因果矩阵
addpath(genpath('D:\qq文件\交接代码\brainstorm\brainstorm3/'));
data_path = 'D:\qq文件\交接代码\数据集\data\processed\sub-umf001_run01_data.mat';
%addpath('D:\qq文件\交接代码\bids-matlab-main\bids-matlab-main');
%BIDS_path = 'D:\qq文件\交接代码\数据集\data\processed\BIDS_dataset\';%sub-umf001\ses-01\eeg\
%BIDS = bids.layout(BIDS_path);
%BIDS_data = bids.query(BIDS, 'metadata');
load(data_path);
%granger参数
order = 3;
inputs =[];
inputs.nTrials = 1;
inputs.standardize = 1;
inputs.flagFPE = 1;
inputs.lag = 0;
inputs.flagELM = 0;
inputs.rho = 50;
%数据参数
Fsample = data.fsample;%采样率
channels = data.label;%通道名
trial_data = data.trial;%样本数据
onset = data.onset_annotation;%发作时间点
trial_data_length = size(trial_data, 2);%样本数据长度
window_length = 4;%窗长，以秒为单位
step_length = window_length * 0.5;%步长，以秒为单位
start_time = (onset - 1.5 * window_length) * Fsample;%遍历起始时间
num_windows = floor( double(trial_data_length - start_time ) / double(step_length * Fsample));%所需要遍历的窗口数
granger_result = [];%存放格兰杰结果
granger_result.connectivity = {};%联通性矩阵
granger_result.channels = channels;%通道名
granger_result.path = data_path;%数据路径
% 遍历数据
for i = 1:num_windows
    start_index = start_time +(i - 1) * step_length * Fsample + 1;%窗口开始
    end_index = start_index + window_length * Fsample - 1;%窗口结束
    
    % 检查窗口是否足够长
    if end_index <= trial_data_length
        X = trial_data(:,start_index:end_index);%sinks
        Y = trial_data(:,start_index:end_index);%sources
        [connectivity,pValue,connectivityV,pValueV] = bst_granger(X,Y,order,inputs);%调用brainstorm进行计算
        granger_result.connectivity{i} = transpose(connectivity);%转置矩阵，结果为纵轴为sources，横轴为sinks
    end
end
save('umf_granger.mat', 'granger_result');%保存结果
