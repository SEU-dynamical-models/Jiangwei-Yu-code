%本代码计算BIDS数据，通过AIC拐点确定最优格兰杰计算的阶
addpath('D:\qq文件\交接代码\');
addpath('D:\qq文件\交接代码\bids-matlab-main\bids-matlab-main');%读取BIDS数据工具包
addpath('D:\qq文件\交接代码\knee_pt');%查找拐点工具
root_path = "D:\qq文件\交接代码\数据集\data\processed\pt\";
sub_files = dir(fullfile(root_path, 'sub*'));
point_set = {};
for i = 1:numel(sub_files)
    filename = sub_files(i).name;
    BIDS_path = char(root_path + filename + "\BIDS_dataset\");%读取该subject的BIDS数据(string类型必须转为char类型，否则bids.layout报错)
    BIDS = bids.layout(BIDS_path);
    BIDS_data = bids.query(BIDS, 'data');%获取BIDS数据下的文件名
    eeg_file = BIDS_data(endsWith(BIDS_data, '.eeg'));%获取所有eeg文件
    cfg = [];
    sub_point_set = {};
    for j = 1:numel(eeg_file)
        cfg.dataset = eeg_file{j};
        BIDS_raw = ft_preprocessing(cfg);%读取eeg数据，包括trial，通道名，采样率等信息
        data = BIDS_raw.trial{1}(1,:);%取每个受试者每此run的第一个通道的信号
        [Lopt,AIC_data] = akaike2(data,1,10,1);%第一个参数为信号数据，第二个参数为第一个尝试阶，第二参数为最后一个尝试阶，第三个参数为信号个数
        x_axis = 1:10;
        [res_x, knee_point] = knee_pt(AIC_data,x_axis,1);%最后一个参数为强制返回值，如果运行出现问题，则返回空值
%         plot(AIC_data);
        sub_point_set{j} = knee_point;
    end
    point_set{i} = sub_point_set;
end