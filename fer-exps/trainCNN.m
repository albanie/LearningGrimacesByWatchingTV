function [dagnet, stats] = trainCNN(opts)
%TRAINCNN Takes in a struct 'opts' which contains the training
% parameters.

% NOTE: All parameters are set in the experimental config file.

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

% Initialize a CNN dagnet using (pretrained) network
dagnet = initPretrainedNet(opts);

% Load training dataset
imdb = loadImdb(opts, dagnet, 'train');

% --------------------------------------------------------------------
%                                                     Evaluate network
% --------------------------------------------------------------------
trainIdx = find(strcmp(imdb.images.set, 'train'));
valIdx = find(strcmp(imdb.images.set, 'val'));

stats = runDAG(dagnet, ...
    imdb, ...
    @getBatch, ...
    opts.train, ...
    'expDir', opts.expDir, ...
    'train', trainIdx, ...
    'val',  valIdx);
end
