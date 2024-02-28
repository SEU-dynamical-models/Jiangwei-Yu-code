%通过格兰杰因果矩阵，计算两两通道间在所有时间窗下的均值和方差
data_path = 'D:\qq文件\交接代码\brainstorm\brainstorm3\umf_granger.mat';
load(data_path);
channel_num = int32(size(granger_result.channels,2));
all_data = cat(3,granger_result.connectivity{:});
mean_and_var = [];
mean_and_var.var_all = zeros(channel_num,channel_num);
mean_and_var.mean_all = zeros(channel_num,channel_num);
mean_and_var.channels = granger_result.channels;
for i = 1:channel_num
    for j = 1:channel_num
        data_to_cal = squeeze(all_data(i, j, :));
        var = nanvar(data_to_cal);
        mean = nanmean(data_to_cal);
        mean_and_var.var_all(i,j) = var;
        mean_and_var.mean_all(i,j) = mean;
    end
end
save('umf_granger_mean_and_var.mat', 'mean_and_var');%保存结果