function wipeFiles(expDir)

lastEpoch = findLastCheckpoint(expDir);

if ~lastEpoch
	return
end
bestEpoch = findBestCheckpoint(expDir, lastEpoch);
preciousEpochs = [bestEpoch lastEpoch];

removeOtherCheckpoints(expDir, preciousEpochs);

% -------------------------------------------------------------------------
function removeOtherCheckpoints(expDir, preciousEpochs )
% -------------------------------------------------------------------------

list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epochs = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
targets = ~ismember(epochs, preciousEpochs);
files = cellfun(@(x) fullfile(expDir, sprintf('net-epoch-%d.mat', x)), ...
		num2cell(epochs(targets)), 'UniformOutput', false);
cellfun(@(x) delete(x), files)


% -------------------------------------------------------------------------
function bestEpoch = findBestCheckpoint(expDir, lastEpoch)
% -------------------------------------------------------------------------
% handle the different storage structures/error metrics
data = load(fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)));
if isfield(data, 'stats')
    valStats = data.stats.val;
elseif isfield(data, 'info')
    valStats = data.info.val;
else
    error('storage structure not recognised');
end

if isfield(valStats, 'classerror')
    errorMetric = [valStats.classerror];
elseif isfield(valStats, 'top1error')
    errorMetric = [valStats.top1error];
elseif isfield(valStats, 'error')
    errorMetric = [valStats.error(1,:)];
else
    error('error metric not recognised');
end
[~, bestEpoch] = min(errorMetric);


% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(expDir)
% -------------------------------------------------------------------------
list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
%assert(epoch > 0, sprintf(['%s does not contain any checkpoints in ', ...
		  %'the form net-epoch-<number>.mat'], expDir));
