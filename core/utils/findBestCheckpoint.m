function bestEpoch = findBestEpoch(expDir)
% returns the epoch number in which the network
% achieved the highest accuracy on the 
% validation set.

lastEpoch = findLastCheckpoint(expDir);
data = load(fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)));
classError = [data.stats.val.classerror];
[~, bestEpoch] = min(classError);
end