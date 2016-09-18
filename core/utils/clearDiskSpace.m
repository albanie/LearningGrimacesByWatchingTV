function clearDiskSpace(dataset)
%CLEARDISKSPACE removes unnecessary training files
%   CLEARDISKSPACE(dataset) searches through experimental 
%   directories created while processing 'dataset' and 
%   deletes all training files except the best epoch 
%   (according to validation score), the final epoch and 
%   the training chart.

expDirs = getExpDirs(dataset);
cellfun(@(x) wipeFiles(x), expDirs);

fprintf('----------------------- \n');
fprintf('%s directories cleaned: \n', dataset);
fprintf('----------------------- \n');
for i = 1:numel(expDirs)
    tokens = strsplit(expDirs{i}, '/');
    fprintf('(%d): %s\n', i, tokens{end - 2});
end