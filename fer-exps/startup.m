% Add paths to core directory and current subfolders
addpath(genpath('../core')) ;
addpath(genpath('.')) ;

% vlfeat is useful for visualizations, but is not required 
vlFeatDir = '~/coding/libs/vlfeat' ;
run(fullfile(vlFeatDir, 'toolbox', 'vl_setup')) ;

% set your path to matconvnet (included as submodule but feel
% free to change path if you already have a compiled version)
matconvnetDir = '../matconvnet' ;
run(fullfile(matconvnetDir, 'matlab', 'vl_setupnn')) ;

% supress graphical debugger (this is down to your personal preferences)
com.mathworks.services.Prefs.setBooleanPref('EditorGraphicalDebugging', false) ;
