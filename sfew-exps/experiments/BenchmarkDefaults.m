function [opts, bestNet] = BenchmarkDefaults(mode, model, gpus, bn, lr, bs, local, opts)
% Returns default options used in experiments

% add paths to server specific config - note that this will be 
% called from the emotions/<experiment_name> directory
addpath '../'
startup;

% generate the experiment name
experimentName = buildExpName(model, bn, lr);

% set shared options
opts.expType = 'benchmarks';
opts.pretrainedNet = model;
opts.modifyForTask = @modifyForSFEW;
experimentRoot = fullfile(paths.DATA_ROOT, 'emotions', 'sfew-naff');
opts.modelDir = fullfile(paths.DATA_ROOT, 'models', 'matconvnet');
opts.dataDir = fullfile(experimentRoot, 'data');
rootExpPath = fullfile(experimentRoot, 'experiments', opts.expType, ...
    model, experimentName);

% --------------------------------------------------------------------
%                                                   Training the model
% --------------------------------------------------------------------

if strcmp(mode, 'train')
    opts.train.gpus = gpus;
    opts.train.LRflip = true;
    opts.train.continue = true;
    opts.train.numEpochs = 60;
    opts.train.batchSize = bs;
    % define possible learning schedules
    keys = {'e2', 'e3', 'e4', 'log24'};
    values = {0.01, 0.001, 0.0001, logspace(-2, -4, opts.train.numEpochs)};
    learningSchedules = containers.Map(keys, values);
    
    opts.train.learningRate = learningSchedules(lr);
    opts.fineTuningRate = 0.1;
    opts.expDir = fullfile(rootExpPath, 'train');
    opts.imdbPath = fullfile(opts.dataDir, opts.pretrainedNet, 'imdb.mat');
    opts.train.expDir = opts.expDir;
    opts.local = local;
    opts.useBnorm = bn;
    opts % display options
    trainCNN(opts);
end

% --------------------------------------------------------------------
%                                                    Testing the model
% --------------------------------------------------------------------

if strcmp(mode, 'test')
    opts.test.gpus = gpus;
    opts.test.numEpochs = 1;
    opts.test.testMode = true;
    opts.test.batchSize = bs;
    % Load the network from the best epoch of training
    bestEpoch = findBestCheckpoint(fullfile(rootExpPath, 'train'));
    data = load(fullfile(rootExpPath, 'train', ...
        strcat('net-epoch-', num2str(bestEpoch), '.mat')));
    bestNet = data.net;
    opts.test.bestEpoch = bestEpoch;
    opts.expDir = fullfile(rootExpPath, 'test');
    opts.imdbPath = fullfile(opts.dataDir, opts.pretrainedNet, 'imdb_test.mat');
    opts.test.expDir = opts.expDir;
    opts.local = local;
    testCNN(bestNet, opts);
end

% --------------------------------------------------------------------
function experimentName = buildExpName(model, bn, lr)
% --------------------------------------------------------------------
experimentName = model;
if bn
    experimentName = sprintf('%s_bn', experimentName);
end
experimentName = sprintf('%s_%s', experimentName, lr);