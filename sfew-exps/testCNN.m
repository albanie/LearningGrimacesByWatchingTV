function testResults = testCNN(net, opts)
% Takes in a struct 'net' which contains the trained parameters
% and  uses it to make predictions on the test set defined in imdb.

% setup the MatConvNet toolbox and add utils
addpath('../matlab');
addpath('IO');
addpath('dag');
addpath('utils');
addpath('stats');
addpath('loss');
addpath('visualization');
vl_setupnn;

% Parameters are set in a config file.

% load a dagnn object
dagnet = dagnn.DagNN.loadobj(net);

% Load training dataset
imdb_test = loadImdb(opts, dagnet, 'test');

% retrieve the test items
testSet = find(imdb_test.images.set == 3);

% set DAG to testMode
dagnet.mode = 'test';

% finally we can evaluate the network
stats = runDAG(dagnet, ...
    imdb_test, ...
    @getBatch, ...
    opts.test, ...
    'val', ...
    testSet);

% save the test results
testResults = stats.val ;
save(strcat(opts.expDir, '/test-results'), 'testResults') ;
end
