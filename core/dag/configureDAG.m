function opts = configureDAG(varargin)
% Provides a set of reasonable defaults for a DAG
% and overrides any given options.

opts.val = [] ;
opts.train = [] ;
opts.testMode = false;
opts.gpus = [] ;
opts.momentum = 0.9 ;
opts.bestEpoch = 1;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.continue = false ;
opts.prefetch = false;
opts.augment = true;
opts.LRflip = false;
opts.numSubBatches = 1 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.expDir = 'experiment';
opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;

opts = vl_argparse(opts, varargin) ;

opts.modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
opts.modelTestFigPath = fullfile(opts.expDir, 'net-test.pdf') ;
opts.modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));

end
