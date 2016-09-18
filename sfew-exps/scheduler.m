function scheduler(gpus)
% SCHEDULER runs an experiment schedule
%   SCHEDULER(gpus) runs the given list of experiments
%   on the gpus provided

startup;
useBnorm = true;
local = false;

if nargin == 0
    gpus = [];
end

% benchmark('alexnet_imagenet', gpus, useBnorm, 'e2', 256, local);
% benchmark('vgg_m_imagenet', gpus, useBnorm, 'e2', 192, local);
benchmark('vgg_vd_face_fer', gpus, useBnorm, 'e2', 32, local);
% benchmark('resnet50_imagenet', gpus, useBnorm, 'e2', 24, local);

clearDiskSpace('sfew');

% -------------------------------------------------------------------------
function benchmark(model, gpus, useBnorm, lr, bs, local)
% -------------------------------------------------------------------------
% BENCHMARK trains and tests a model
%   BENCHMARK(model, gpus, useBnorm, lr, bs, local) trains and tests 
%   the given model using the provided parameters
%   
%   'useBnorm' (boolean): train the network with Batch Normalization
%   'gpus' (Array): a list of gpu devices to be used
%   'lr' (scalar/Array): learning rate schedule
%   'bs' (scalar): batch size
%   'local' (boolean): run the experiment "locally" with only a small
%       subset of the training data
BenchmarkDefaults('train', model, gpus, useBnorm, lr, bs, local);
BenchmarkDefaults('test', model, gpus, useBnorm, lr, bs, local);
