function benchmark_fer_models(varargin)
%BENCHMARK_FER_MODELS - evaluate trained models on FER
%   BENCHMARK_FER_MODELS - evaluates the performance of trained emotion
%   recognition CNNs on the FER validation and test sets.
%
%   BENCHMARK_FER_MODELS(..'name', value) accepts the following
%   options:
%
%   `refresh` :: false
%    If true, refreshes any cached results from previous evaluations.
%
%   `benchmarkCacheDir` :: fullfile(vl_rootnn, 'data/grimaces/benchCache')
%    Directory where evaluation results for each model will be cached.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.refresh = false ;
  opts.benchmarkCacheDir = fullfile(vl_rootnn, 'data/grimaces/benchCache') ;
  opts = vl_argparse(opts, varargin) ;

  if ~exist(opts.benchmarkCacheDir, 'dir')
    mkdir(opts.benchmarkCacheDir) ;
  end

  pretrained = {
    'alexnet-face-fer-bn', ...
    'vgg-m-face-bn-fer', ...
    'vgg-vd-face-fer', ...
	} ;

  for ii = 1:numel(pretrained)
    fprintf('-----------------------------------------------------------\n') ;
    modelName = pretrained{ii} ;
    cachePath = fullfile(opts.benchmarkCacheDir, ...
                         sprintf('%s.mat', modelName)) ;
    if exist(cachePath, 'file') && ~opts.refresh
      fprintf('loading cached results for %s on FER...\n', modelName) ;
      stats = load(cachePath) ;
    else
      fprintf('evaluating %s on FER...\n', modelName) ;
      [~,valInfo] = fer_baselines('modelName', modelName, ...
                               'evaluateOnly.subset', 'val', ...
                               'train.batchSize', 32) ;
      [~,testInfo] = fer_baselines('modelName', modelName, ...
                               'evaluateOnly.subset', 'test', ...
                               'train.batchSize', 32) ;
      valAcc = 1 - valInfo.val.classerror ;
      testAcc = 1 - testInfo.val.classerror ;
      stats.valAcc = valAcc ; stats.testAcc = testAcc ;
      fprintf('caching results for %s to %s ...', modelName, cachePath) ; tic ;
      save(cachePath, '-struct', 'stats') ;
      fprintf('done in %g s \n', toc) ;
    end
    fprintf('%s: val (%.3f), test: (%.3f)\n', ...
                         modelName, stats.valAcc, stats.testAcc) ;
	end
