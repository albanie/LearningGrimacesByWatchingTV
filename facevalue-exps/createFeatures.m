function createFeatures(gpus)

% setup matconvnet
run('../configure_server.m');
vl_setupnn;

opts.batchSize = 32;
model = 'vgg-m-face-bn';
opts.dataDir = fullfile(DATA_ROOT, 'fer', 'data');
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
modelPath = fullfile(DATA_ROOT, 'models', strcat(model,'.mat'));
opts.expDir = fullfile(DATA_ROOT, 'xdondfer', 'experiments', model);
dagnet = dagnn.DagNN.fromSimpleNN(load(modelPath), 'canonicalNames', true);

opts.gpus = gpus;
opts.local = true;
% load data
imdb = loadImdb(opts, dagnet, 'train');
prediction_scores = zeros(numel(imdb.images.labels), 2622);
fc_scores = zeros(numel(imdb.images.labels), 4096);

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  dagnet.move('gpu') ;
end

for t = 1:opts.batchSize:50%numel(imdb.images.labels);
    
    % print progress to screen
    fprintf('Progress:  %3d/%3d\n',  ...
        fix((t-1)/opts.batchSize)+1, ...
        ceil(numel(imdb.images.labels)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(imdb.images.labels) - t + 1) ;
    
    % get this image batch and prefetch the next
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1, numel(imdb.images.labels)) ;
    batch = batchStart : batchEnd ;
    
    % once all data has been processed, break out
    if numel(batch) == 0, continue ; end
    
    % load the images and labels
    inputs = getBatch(imdb, batch, 'test', opts, dagnet) ;
    
    % store the values at the feature layer
    dagnet.vars(dagnet.getVarIndex('x24')).precious = true ;
    dagnet.vars(dagnet.getVarIndex('prediction')).precious = true ;
    
    % retrieve the scores
    dagnet.mode = 'test';
    dagnet.eval({inputs{1:2}});
    
    % obtain the CNN otuput
    prediction_score = dagnet.vars(dagnet.getVarIndex('prediction')).value ;
    prediction_score = squeeze(gather(prediction_score)) ;
    
    fc_score = dagnet.vars(dagnet.getVarIndex('x24')).value ;
    fc_score = squeeze(gather(fc_score)) ;
    
    % show the classification results
%     [bestScore, best] = max(scores) ;
    prediction_scores(batch,:) = prediction_score';
    fc_scores(batch,:) = fc_score';
end

dagnet.reset() ;
dagnet.move('cpu') ;

mkdir(opts.expDir);
% results.prediction_scores = single(prediction_scores);
results.fc_scores = single(fc_scores);
results.boxLabels = uint8(imdb.images.labels);
results.set = uint8(imdb.images.set);
save('-v7.3', fullfile(opts.expDir, 'results'), '-struct', 'results');
