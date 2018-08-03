function [net, info] = cnn_emo_baselines(varargin)
% CNN_EMO_BASELINES train simple baselines on emotion recognition
%   [NET, INFO] = CNN_EMO_BASELINES(VARARGIN)
%
%   CNN_EMO_BASELINES(..'name', value) accepts the following
%   options:
%
%   `dropoutRate` :: 0.5
%    If dropoutRate > 0, dropout layers with the given rate are inserted
%    into the model.
%
%   `dataAug` :: true
%    Whether to apply data augmentation during training.
%
%   `useBnorm` :: true
%    If true, inserts batch norm layers into a model after
%    each convolution. If the model already contains any batch norm layers,
%    this option does nothing.
%
%   `numOutputs` :: 7
%    The number of output classes to be predicted by the model (FER has seven
%    but some variants of this dataset use different numbers).
%
%   `modelName` :: `vgg-vd-face'
%    The architecture used as a basis for the emotion recognition CNN.
%
%   `dataDir` :: fullfile(vl_rootnn, 'data/datasets/fer2013')
%    The directory containing the dataset data (i.e. the fer2013.csv file).
%
%   The following options are used for when only evaluation (not training) is
%   required:
%
%   `evaluateOnly.subset` :: ''
%    If 'val' or 'test', evaluates the best availabale checkpoint with the given
%    options, on the specified subset of the data, but does not continue
%    training.
%
%   `evaluateOnly.fromCkpt` :: false
%    If true, searches experiment directories with the given configuration
%    to find the best checkpoint to evaluate. If false, directly evaluates
%    the model with the given MODELNAME.
%
%   This example is based on the `cnn_cifar.m` script provided in MatConvNet.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.dev = false ;
  opts.useBnorm = 1 ;
  opts.dataAug = true ;
  opts.numOutputs = 7 ;
  opts.finetuneLR = 0.1 ;
  opts.dropoutRate = 0.5 ;
  opts.dataset = 'fer2013' ;
  opts.modelName = 'vgg-vd-face-fer' ;
  opts.dataDir = fullfile(vl_rootnn, 'data/datasets/fer2013') ;

  opts.evaluateOnly.subset = '' ;
  opts.evaluateOnly.fromCkpt = false ;

  opts.train.gpus = 1 ;
  opts.train.batchSize = 128 ;
  opts.train.continue = true ;
  opts.train.extractStatsFn = @extractStats ;
  opts.train.learningRate = [ones(1, 40) * 0.01 ...
                             ones(1, 10) * 0.001 ...
                             ones(1, 10) * 0.0001] ;
  opts = vl_argparse(opts, varargin) ;

  expDir = buildExpDirName(opts) ;
  opts.imdbPath = fullfile(vl_rootnn, 'data/grimaces', ...
                           opts.dataset, 'imdb.mat') ;

  fprintf('loading network...') ; tic ;
  dag = grimaces_zoo(opts.modelName, 'useBnorm', opts.useBnorm, ...
                                     'finetuneLR', opts.finetuneLR, ...
                                     'dropoutRate', opts.dropoutRate, ...
                                     'numOutputs', opts.numOutputs) ;
  fprintf('done in %g s\n', toc) ;

  switch opts.dataset
    case 'sfew', getImdb = @getSfewImdb ;
    case 'fer2013', getImdb = @getFerImdb ;
    otherwise, error('unrecognised dataset: %s\n', opts.dataset) ;
  end

  fprintf('loading imdb...') ; tic ;
  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
  else
    imdb = getImdb(opts.dataDir) ;
    mkdir(fileparts(opts.imdbPath)) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
  end
  fprintf('done in %g s\n', toc) ;

  if opts.dev % dev-only
    sample = 1000 ;
    imdb.images.set = (imdb.images.set * 0) + 4 ;
    imdb.images.set(1:sample) = 1 ;
    imdb.images.set(sample+1:2 * sample) = 2 ;
    opts.train.numEpochs = 1 ;
  end

  if ismember(opts.evaluateOnly.subset, {'val', 'test'})
    if opts.evaluateOnly.fromCkpt % check for existing checkpoints
      best = findBestEpoch(expDir, 'priorityMetric', 'classerror', ...
                                    'prune', true) ;
      ckptPath = fullfile(expDir, sprintf('net-epoch-%d.mat', best)) ;
      fprintf('loading from checkpointing %s...\n', ckptPath) ;
      tmp = load(ckptPath) ; dag = dagnn.DagNN.loadobj(tmp.net) ;
    end
    imdb.images.set(imdb.images.set == 1) = 4 ;
    if strcmp(opts.evaluateOnly.subset, 'test')
      % ensure that evaluation takes place on the desired subset
      imdb.images.set(imdb.images.set == 2) = 4 ;
      imdb.images.set(imdb.images.set == 3) = 2 ;
    end
    opts.train.numEpochs = 1 ;
    opts.train.continue = 0 ;
  end

  getBatchFn = @(i,b) getBatch(i, b, opts, dag) ;
  dag.meta.classes.name = imdb.meta.classes(:)' ;
  [net, info] = cnn_train_dag(dag, imdb, getBatchFn, ...
                              opts.train, 'expDir', expDir) ;

% -----------------------------------------------------------
function expDir = buildExpDirName(opts)
% -----------------------------------------------------------
  expDir = fullfile(vl_rootnn, 'data/grimaces', opts.dataset, opts.modelName) ;
  if opts.dropoutRate > 0
    expDir = [expDir sprintf('-dropout-%g', opts.dropoutRate)] ;
  end
  if opts.dataAug
    expDir = [expDir '-aug'] ;
  end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
  sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
  for ii = 1:numel(sel)
    target = sel(ii) ;
    if net.layers(target).block.ignoreAverage, continue ; end
    stats.(net.layers(target).outputs{1}) = net.layers(target).block.average ;
  end

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch, opts, dag)
% --------------------------------------------------------------------
% GETBATCH- load a batch of data from the current IMDB
%    INPUTS = GETBATCH(IMDB, BATCH, OPTS, DAG) returns the inputs and
%    associated labels in the batch specified by the indices in BATCH by
%    finding them in the IMDB data structure.

  labels = imdb.images.labels(1, batch) ;

  % ensure that all instances come form a single set
  setIdx = unique(imdb.images.set(batch)) ;
  assert(numel(setIdx) == 1, ...
             'training/val/test sets have gotten mixed together!') ;
  trainMode = setIdx == 1 ; % check whether we should apply data aug

  % Note that the raw data for FER is greyscale
  greyData = imdb.images.data(:,:,:,batch) ; sz = size(greyData) ;
  data = zeros(sz(1), sz(2), 3, numel(batch), 'single') ; % will store as RGB

  for idx = 1:numel(batch)
    img = greyData(:,:,:,idx) ;
    img = normalizeFace(img, dag) ; % normalize face (grey -> RGB)
    if trainMode
      if rand > 0.5, img = fliplr(img) ; end
    end
    data(:,:,:,idx) = img ;
  end
  if numel(opts.train.gpus) > 0, data = gpuArray(data) ; end

  augs = computeAugs(numel(batch)) ;
  transforms = zeros(1, 1, 6, numel(batch)) ;
  for i = 1:numel(batch)
    if opts.dataAug & trainMode
      tmp = augs(:,:,i) ;
    else
      % NOTE: even when data augmentation is not being used, it is still
      % faster to use the bilinearsampler on the GPU to resize the images
      % than to use the builtin MATLAB imresize function.
      tmp = [1 0 0 ;
             0 1 0 ;
             0 0 1] ;
    end
    transforms(:,:,:,i) = tmp([5 4 2 1 8 7]) ; % re-order to match interface
  end
  grid = vl_nnaffinegrid(transforms, dag.meta.normalization.imageSize(1:2)) ;
  if numel(opts.train.gpus) > 0
    grid = gpuArray(single(grid)) ;
  end
  data = vl_nnbilinearsampler(data, grid) ;
  inputs = {'data', data, 'label', labels} ;

% --------------------------------------------------------------------
function face = normalizeFace(greyFace, dag)
% --------------------------------------------------------------------
  % handle average image consisting of color channel averages
  face = repmat(greyFace, [1 1 3]) ;
  avgIm = reshape(dag.meta.normalization.averageImage, 1, 1, 3) ;
  face = bsxfun(@minus, face, avgIm) ;

% --------------------------------------------------------------------
function affs = computeAugs(batchSize)
% --------------------------------------------------------------------
%COMPUTEAUGS - compute augmentation matrices
%     AFFS = COMPUTEAUGS(BATCHSIZE) - computes a collection of affine
%     transformations corresponding to various augmentations of the data.
%     These augmentation transformations are based on the following paper:
%
%        Yu, Zhiding, and Cha Zhang. "Image based static facial expression
%        recognition with multiple deep network learning." ACM-MI, 2015.

  ratio = 1/25 ;
  maxOffset = round(ratio * 224) ;

  minXY = randi(maxOffset, batchSize, 2) ;
  zoomSc = (1 - ratio) + (ratio*2) * rand(1, batchSize) ;
  zAffs = arrayfun(@(x) {zoomOut(zoomSc(x), minXY(x,:))}, 1:batchSize) ;
  zAffs = cat(3, zAffs{:}) ;

  vals = [-pi/18 0 pi/18] ;
  thetas = randi(3, batchSize) ;
  rAffs = arrayfun(@(x) {rotate(vals(thetas(x)))}, 1:batchSize) ;
  rAffs = cat(3, rAffs{:}) ;

  vals = [-0.1 0 0.1] ;
  skews = randi(3, batchSize, 2) ;
  sAffs = arrayfun(@(x) {skew(vals(skews(x,1)), vals(skews(x,2)))}, 1:batchSize) ;
  sAffs = cat(3, sAffs{:}) ;

  for ii = 1:batchSize
    affs(:,:,ii) = zAffs(:,:,ii) * rAffs(:,:,ii) * sAffs(:,:,ii) ; %#ok
  end

  % only apply data augmentation 50% of the time
  drop = find(rand(1, batchSize) > 0.5) ;
  for ii = 1:numel(drop)
    affs(:,:,drop(ii)) = eye(3,3) ;
  end

% --------------------------------------------------------
function aff = zoomOut(zoomScale, minYX)
% --------------------------------------------------------
	zs = (zoomScale - 1) / zoomScale ;
	tx = zs - 2 * zs * minYX(2) ;
	ty = zs - 2 * zs * minYX(1) ;
	aff = [ 1 0 tx ; % compute the affine matrix
					0 1 ty ;
					0 0 1] * zoomScale ;

% --------------------------------------------------------
function aff = rotate(theta)
% --------------------------------------------------------
% compute the affine matrix for a rotation with angle THETA
	aff = [ cos(theta) -sin(theta) 0 ;
					sin(theta) cos(theta) 0 ;
					0 0 1] ;

% --------------------------------------------------------
function aff = skew(s1, s2)
% --------------------------------------------------------
% compute the affine matrix for a skew with parameters S1 and S2
	aff = [ 1 s1 0 ;
					s2 1 0 ;
					0 0 1] ;
