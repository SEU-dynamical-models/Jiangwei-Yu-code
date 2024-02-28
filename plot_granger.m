%通过计算得到的均值和方差，画出热力图图像
data_path = 'D:\qq文件\交接代码\brainstorm\brainstorm3\umf_granger_mean_and_var.mat';
load(data_path);
var = mean_and_var.var_all;
mean = mean_and_var.mean_all;
channels = mean_and_var.channels;
figure;
imagesc(var);
colorbar; % 添加颜色条，以显示颜色与数值的对应关系
title('Var Plot');
xticks(1:size(channels,2));
yticks(1:size(channels,2));
xticklabels(channels);
yticklabels(channels);
figure;
imagesc(mean);
colorbar;
title('Mean Plot');
xticks(1:size(channels,2));
yticks(1:size(channels,2));
xticklabels(channels);
yticklabels(channels);