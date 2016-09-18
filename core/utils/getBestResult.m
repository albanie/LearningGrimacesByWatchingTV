function result = getBestResult(expDir)
%GETBESTRESULT Returns best result from an experiment directory.
%   RESULT = GETBESTRESULT(dataset) uses
%   validation accuracy to select the best epoch of network 
%   training. It returns the experiment name, the accuracy from 
%   the best epoch, and the epoch number of the best epoch.

expParts = strsplit(expDir,'/');
result.expName = expParts{end - 1};
lastEpoch = findLastCheckpoint(expDir);
[valAccuracy, bestEpoch] = getBestCheckpoint(expDir, lastEpoch);
testAccuracy = getTestAccuracy(expDir);
result.bestEpoch= bestEpoch;
result.valAccuracy = valAccuracy;
result.testAccuracy = testAccuracy;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(expDir)
% -------------------------------------------------------------------------

list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function [accuracy, bestEpoch] = getBestCheckpoint(expDir, lastEpoch)
% -------------------------------------------------------------------------

if ~lastEpoch
    [accuracy, bestEpoch] = deal('NA');    
	return
end

data = load(fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)));
classError = [data.stats.val.classerror];
[minClassError, bestEpoch] = min(classError);
accuracy = 1 - minClassError;

% format values for easy readability
bestEpoch = num2str(bestEpoch);
accuracy = sprintf('%0.3f', accuracy * 100);

% -------------------------------------------------------------------------
function accuracy = getTestAccuracy(expDir)
% -------------------------------------------------------------------------

testResultsPath = fullfile(fileparts(expDir), 'test', 'test-results.mat');
if exist(testResultsPath, 'file')
    data = load(testResultsPath);
    accuracy = 1 - data.testResults.classerror;
    accuracy = num2str(accuracy);
else
    accuracy = 'NA'; 
end

