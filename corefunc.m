addpath('D:\qq文件\交接代码\数据集\effective_connectivity\TRENTOOL3-3.4.2\TRENTOOL3-master/')
addpath('D:\qq文件\交接代码\数据集\effective_connectivity\TRENTOOL3-3.4.2\fieldtrip-master/');
ft_defaults;
cd('D:\qq文件\交接代码\数据集\effective_connectivity\TRENTOOL3-3.4.2\TRENTOOL3_exampledata-master\Lorenz_3_systems');
myOutputpath = 'D:\qq文件\交接代码\数据集\effective_connectivity\TRENTOOL3-3.4.2\TRENTOOL3_exampledata-master\lorenz_3_output';
load('lorenz_1-2-3_delay_20_20_ms.mat');
dataType = class(data.fsample);    % 使用class函数获取变量x的数据类型
disp(dataType);         % 输出结果为'double'
%% define cfg for TEprepare.m

cfgTEP = [];


% data
cfgTEP.toi                 = [data.time{1}(1) data.time{1}(end)]; % time of interest
cfgTEP.channel              = data.label;  % channels to be analyzed

% ensemble methode
cfgTEP.ensemblemethod = 'no';

% estimator
cfgTEP.TEcalctype  = 'VW_ds'; % use the new TE estimator (Wibral, 2013)

% scanning of interaction delays u
cfgTEP.predicttime_u  = 20;      % minimum u to be scanned

% ACT estimation and constraints on allowed ACT(autocorelation time)
cfgTEP.actthrvalue = 100;   % threshold for ACT
cfgTEP.maxlag      = 1000;
cfgTEP.minnrtrials = 15;   % minimum acceptable number of trials

% optimizing embedding
cfgTEP.optimizemethod ='ragwitz';  % criterion used
cfgTEP.ragdim         = 2:9;       % criterion dimension
cfgTEP.ragtaurange    = [0.2 0.4]; % range for tau
cfgTEP.ragtausteps    = 5;        % steps for ragwitz tau steps
cfgTEP.repPred        = 100;      % size(data.trial{1,1},2)*(3/4);

% kernel-based TE estimation
cfgTEP.flagNei = 'Mass' ;           % neigbour analyse type
cfgTEP.sizeNei = 4;                 % neigbours to analyse

% set the level of verbosity of console outputs
cfgTEP.verbosity = 'info_minor';

%% define cfg for TEsurrogatestats_ensemble.m

cfgTESS = [];

% use individual dimensions for embedding
cfgTESS.optdimusage = 'indivdim';
cfgTESS.embedsource = 'no';

% statistical and shift testing
cfgTESS.tail           = 1;
cfgTESS.numpermutation = 5e4;
cfgTESS.surrogatetype  = 'trialshuffling';

% shift test
cfgTESS.shifttest      = 'no';      % don't test for volume conduction

% results file name
cfgTESS.fileidout  = strcat(myOutputpath,'/Lorenzdata_3_');

%% calculation - scan over specified values for u

data_prepared = TEprepare(cfgTEP,data);
TEpermtest = TEsurrogatestats(cfgTESS,data_prepared);

save([cfgTESS.fileidout 'Lorenz_3_TEpermtest_output.mat'],'TEpermtest');

