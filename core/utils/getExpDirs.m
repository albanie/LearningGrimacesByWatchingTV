function expDirs = getExpDirs(dataset)
%GETEXPDIRS Returns a cell array of experiment directories.
%   EXPDIRS = GETEXPDIRS(dataset) recursviely searches the 
%   directory tree of the set of experiments specified by
%   'dataset' and finds any folders containing experiment 
%   files in the form 'net-epoch-<num>.mat'.  These folders 
%   are returned as a cell array.

% add paths to server specific config 
addpath '../';
configure_server;

% check for benchmark/pretraining experiments
targetDirs = {'benchmarks', 'pretraining'};
rootDir = fullfile(DATA_ROOT, dataset, 'experiments');
expDirs = cellfun(@(x) getExpDirsRecursively(...
                fullfile(rootDir, x)), ...
                targetDirs, ...
                'UniformOutput', false);

expDirs = horzcat(expDirs{:});            


function expDirs = getExpDirsRecursively(rootDir)
% searches for experimental files recursively from the given
% root directory

% Get a list of the model names
files = ignoreSystemFiles(dir(rootDir));
modelNames = {files.name};

expDirs = {};
for i = 1:numel(modelNames)
	model = modelNames{i};
	modelFiles = ignoreSystemFiles(dir(fullfile(rootDir, model)));
	modelFiles = {modelFiles.name};
	for j = 1:numel(modelFiles)
		expDirs{end + 1} = fullfile(rootDir, modelNames{i}, modelFiles{j}, 'train');
	end
end
