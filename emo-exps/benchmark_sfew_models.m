function benchmark_sfew_models(varargin)
%BENCHMARK_SFEW_MODELS - evaluate trained models on SFEW
%   BENCHMARK_SFEW_MODELS - evaluates the performance of trained emotion
%   recognition CNNs on the SFEW validation and test sets.
%
%   BENCHMARK_SFEW_MODELS(..'name', value) accepts the following
%   options:
%
%   `refresh` :: false
%    If true, refreshes any cached results from previous evaluations.
%
%   `benchmarkCacheDir` :: fullfile(vl_rootnn, 'data/grimaces/sfew/benchCache')
%    Directory where evaluation results for each model will be cached.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.refresh = false ;
  opts.benchmarkCacheDir = fullfile(vl_rootnn, 'data/grimaces/sfew/benchCache') ;
  opts = vl_argparse(opts, varargin) ;

  if ~exist(opts.benchmarkCacheDir, 'dir')
    mkdir(opts.benchmarkCacheDir) ;
  end

  % These models have not been finetuned on SFEW, but they have been trained
  % for emotion recognition so its interesting to see how they perform.
  ferModels = {
    'alexnet-face-fer-bn', ...
    'vgg-m-face-bn-fer', ...
    'vgg-vd-face-fer', ...
  } ;

  pretrained = [{
    'vgg-vd-face-sfew', ...
    'resnet50-face-sfew', ...
	} ferModels] ;

  for ii = 1:numel(pretrained)
    fprintf('-----------------------------------------------------------\n') ;
    modelName = pretrained{ii} ;
    cachePath = fullfile(opts.benchmarkCacheDir, ...
                         sprintf('%s.mat', modelName)) ;
    if exist(cachePath, 'file') && ~opts.refresh
      fprintf('loading cached results for %s on SFEW...\n', modelName) ;
      stats = load(cachePath) ;
    else
      commonArgs = {'modelName', modelName, ...
                    'train.batchSize', 32, ...
                    'dataset', 'sfew', ...
                    'dataDir', fullfile(vl_rootnn, 'data/datasets/sfew')} ;
      fprintf('evaluating %s on SFEW...\n', modelName) ;
      [~,valInfo] = cnn_emo_baselines('evaluateOnly.subset', 'val', ...
                                  commonArgs{:}) ;
      valAcc = 1 - valInfo.val.classerror ;
      stats.valAcc = valAcc ;
      fprintf('caching results for %s to %s ...', modelName, cachePath) ; tic ;
      save(cachePath, '-struct', 'stats') ;
      fprintf('done in %g s \n', toc) ;
    end
    fprintf('%s: val (%.3f)\n', modelName, stats.valAcc) ;
	end
