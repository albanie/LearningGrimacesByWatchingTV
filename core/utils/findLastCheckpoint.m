function epoch = findLastCheckpoint(expDir)
% returns the epoch number of the most up to date 
% saved state of training.

fileList = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({fileList.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]);
end