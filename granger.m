%计算格兰杰因果矩阵

addpath(genpath('D:\qq文件\交接代码\brainstorm\brainstorm3/'));%通过brainstorm计算格兰杰因果
addpath('D:\qq文件\交接代码\bids-matlab-main\bids-matlab-main');%读取BIDS数据工具包
root_path = "D:\qq文件\交接代码\数据集\data\processed\pt\";%"/home/yujiangwei/processed_open_data/jh";
sub_files = dir(fullfile(root_path, 'sub*'));
%granger参数
inputs =[];
inputs.nTrials = 1;
inputs.standardize = 1;
inputs.flagFPE = 1;
inputs.lag = 0;
inputs.flagELM = 0;
inputs.rho = 50;
%遍历所有受试者
for j = 1:numel(sub_files)
    filename = sub_files(j).name;
    BIDS_path = char(root_path + filename + "\BIDS_dataset\");%读取该subject的BIDS数据(string类型必须转为char类型，否则bids.layout报错)
    BIDS = bids.layout(BIDS_path);
    BIDS_data = bids.query(BIDS, 'data');%获取BIDS数据下的文件名
    eeg_file = BIDS_data(endsWith(BIDS_data, '.eeg'));%获取所有eeg文件
    vmrk_file = BIDS_data(endsWith(BIDS_data, '.vmrk'));%获取所有vmrk文件
    for k = 1:numel(eeg_file)%遍历所有run（eeg和vmrk数据数量应该相同）
        cfg = [];
        cfg.dataset = eeg_file{k};
        event = ft_read_event(vmrk_file{k});%读取vmrk的event数据，得到onset
        BIDS_raw = ft_preprocessing(cfg);%读取eeg数据，包括trial，通道名，采样率等信息
        %获取最佳阶数
        AIC_data = load('D:\qq文件\交接代码\数据集\data\processed\pt_AIC.mat');
        best_order = AIC_data.point_set{j}{k};%获取第j个受试者的第k次run的最佳阶
        %数据参数
        Fsample = BIDS_raw.fsample;%采样率
        channels = BIDS_raw.label;%通道名
        trial_data = BIDS_raw.trial{1};%样本数据
        onset = event.sample;%发作时间点样本
        trial_data_length = size(trial_data, 2);%样本数据长度
        window_length = 4;%窗长，以秒为单位
        step_length = window_length * 0.5;%步长，以秒为单位
        start_time = onset - 1.5 * window_length * Fsample;%遍历起始时间
        num_windows = floor( double(trial_data_length - start_time ) / double(step_length * Fsample));%所需要遍历的窗口数
        granger_result = [];%存放格兰杰结果
        granger_result.connectivity = {};%联通性矩阵
        granger_result.channels = channels;%通道名
        % 遍历数据
        for i = 1:num_windows
            start_index = start_time +(i - 1) * step_length * Fsample + 1;%窗口开始
            end_index = start_index + window_length * Fsample - 1;%窗口结束

            % 检查窗口是否足够长
            if end_index <= trial_data_length
                X = trial_data(:,start_index:end_index);%sinks
                Y = trial_data(:,start_index:end_index);%sources
                [connectivity,pValue,connectivityV,pValueV] = bst_granger(X,Y,best_order,inputs);%调用brainstorm进行计算
                connectivity(1:size(connectivity,1)+1:end) = 0;%联通性矩阵对角线置零
                granger_result.connectivity{i} = transpose(connectivity);%转置矩阵，结果为纵轴为sources，横轴为sinks
            end
        end
        granger_save_path = "pt_"+ filename + "_run" + num2str(k) + "_granger.mat";
        save(granger_save_path, 'granger_result');%保存结果
    end
end
