% setup matconvnet
addpath('~/coding/src/corelib/utils');

paths = loadPaths();
addpath(genpath(paths.CORE));
addpath(genpath('.'));
run(fullfile(paths.VLFEAT, 'toolbox', 'vl_setup'));
run(fullfile(paths.MATCONVNET, 'matlab', 'vl_setupnn'));

% add experiments to path
addpath(genpath('experiments'))

% add core code to path
% addpath(genpath('../core/'));

% add path to utils
addpath('utils');

