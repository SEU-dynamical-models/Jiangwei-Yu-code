% ------------------------------------------------------------------------%
% function ModelRun(indir, outdir)
% note: ModelRun(indir) works as ModelRun(indir, indir)
% ------------------------------------------------------------------------%
% inputs:
% indir  : a subdirectory of ./net_results
% outdir : a subdirectory of ./analysis_results
% indir contains the config file and the functional connectivity results
% outdir will contain files with the same names as in ./net_results/indir
% but with the results of the BNI and NI analysis with the theta model
% ------------------------------------------------------------------------%
% outputs:
% For each functional connectivity network in the directory
% ./net_results/indir, there is an output
% For each patient/seizure file, there is a file in ./net_results/indir
% with a network for all epochs, frequency bands, connectivity method
% For each patient/seizure file, this will create a file in
% ./net_results/outdir with the analysis of each network
%
% input is of the form a.CC(:,:,1:N) 'a','b'... -freq band, 'CC','MI' ...
% - conn method, N - number of epochs for that 'pat' (see config.ini)
% output is of the from output.a.CC{1} ... a.CC{N} with the analysis for
% that network
% ------------------------------------------------------------------------%
% For each epoch:
% BNI_find.m - a list of attempted weights w and the resulting BNI for it
%          - it continues until it find a w such that BNI ~ 0.5
% BNI_single.m - finds the final w, which interpolates the result of BNI_find
% to try and optimize w such that BNI = 0.5
% NI_model.m - given that w, it removes each node one at a time and
% recalculates BNI. Ictogenic nodes lower BNI when removed
% analysis.m - see ./MEGmat/analysis.mat, but this rank orders the nodes in a
% couple ways and saves a few other interesting results
% ------------------------------------------------------------------------%

% All extra functions are in ./MEGmat
addpath('D:\qq文件\交接代码\ViEEG-main\ViEEG-main\modelling\MEGmat');
% addpath('/home/yujiangwei/MEGmat/');

delete(gcp('nocreate'));

% Activate parallel pool (with max number of workers on the machine)
myCluster = parcluster('local');
parpool('local', myCluster.NumWorkers);

% seed the random number generator (seed is saved in the output .mat
% file)
rng('shuffle');
seed = randi(2^32-1);
rng(seed);

% Inputs will be in ./net_results/indir
% Outputs will be in ./analysis_results/outdir (but usually we just
% want this to be in ./analysis_results/indir so allow only one input
% to ModelRun(indir) = ModelRun(indir, indir)
% try
%     outdir = fullfile('analysis_results',outdir);
% catch
%     outdir = fullfile('analysis_results',indir);
% end
% indir = fullfile('net_results',indir);

% Iterate over all Patient/seizure files in ./net_results/indir
% files = dir(fullfile(indir,'Patient*.mat'));


% parameters for running the theta model (BNI, NI)
params.T=4*10^6;         % # time steps
params.n_n=8;            % # runs for noise
params.I_0=-1.2;         % distance to SNIC
params.I_sig=5*1.2*0.1;  % noise level


% connectivity matrix for the given time segment,
% functional connectivity method, frequency band,
% patient/seizure file

%outdir = '/home/yujiangwei/Theta_model_result/umf/';
root_path = "D:\qq文件\交接代码\数据集\data\processed\ummc_granger";
sub_files = dir(fullfile(root_path, 'ummc*'));
for i = 1:numel(sub_files)
    filename = sub_files(i).name;
    data = load('/home/yujiangwei/granger_data/umf/umf_sub1_run1_granger.mat');
    nums = size(data.granger_result.connectivity,2);
    for k=1:nums
        network = data.granger_result.connectivity{k};
    %     count = 1;
    %     network1 = squeeze(data(k,:,:));
    %     network = zeros(chl,chl);
    %     for i=1:chl
    %         for j=1:chl
    %             temp = network1(i,j) - network1(j,i);
    %             if(temp < 0)
    %                 network(i,j)=0;
    %             else
    %                 network(i,j) = temp;
    %             end
    %         end
    %     end
    %    network = squeeze(data(k,:,:));
    %     network = network';
    %     network = (network + network')/2;
        count = 1;

        % We will need this temp file to create the output struct
        temp = {};temp.out = {};temp.out_full = {};
        temp.NI_out = {}; temp.NI_out_full = {};temp.results = {};

        % Format output of the BNI, NI modelling
        % output.(fn{k}).(fn2{k2}) = temp;
        out = {}; out_full = {}; NI_out = {}; NI_out_full = {}; results = {};
        % Iterate over analysed time segments

        % Find the weight w, such the the connectivity matrix
        % gives a BNI = 0.5. This finds a close w value on
        % either side
        [out, out_full] = BNI_find(network, out, out_full, count, params);
        % Linear interpolation of the results from BNI_find to
        % find a weight w that is hopefully very close to
        % giving BNI = 0.5
        [out, out_full] = BNI_single(network, out, out_full, count, params);

        % Calculate NI. Remove each source and recalculate BNI
        % to see if it increases or decreases
        [NI_out, NI_out_full] = NI_model(network, out, NI_out, NI_out_full, count, params);
        % Rank order the nodes in terms of ictogeniticy and
        % find the significantly ictogenic nodes (see analysis
        % for all the fun stuff it does)
        [results] = analysis(results, out_full, NI_out_full, count);

        % Put the result in output, which has a nearly identical
        % form to the input in ./net_results/indir
        % output.a.CC{1} is modelling results for in.a.CC(:,:,1)
        % output.b.CC{2} is modelling results for in.b.CC(:,:,2)
        % etc.
        output.out = out;
        output.out_full = out_full;
        output.NI_out = NI_out;
        output.NI_out_full = NI_out_full;
        output.results = results;
        % save the seed as well for posterity
        % as with the network results, each Patient/Seizure in
        % 'pats' has it's own file
        time_file = "tmo_umf_window"+ num2str(k)+"_granger.mat";
        save(fullfile(outdir, time_file),'output','seed','params');
    end
end
