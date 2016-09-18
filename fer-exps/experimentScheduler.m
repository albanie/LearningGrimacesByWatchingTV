function experimentScheduler(gpus)
%EXPERIMENTSCHEDULER runs the given list of experiments
%   on the gpus provided

addpath('experiments');

if nargin == 0
    gpus = [];
end

useBnorm = true;
local = true;

% imagenet-pretrained models
benchmark('alexnet_imagenet', gpus, useBnorm, 'e2', 256, local);
benchmark('vgg_m_imagenet', gpus, useBnorm, 'e2', 192, local);
benchmark('vgg_vd_imagenet', gpus, useBnorm, 'e2', 32, local);
benchmark('resnet50_imagenet', gpus, useBnorm, 'e2', 24, local);

% VGGFace-pretrained models
benchmark('alexnet_face', gpus, useBnorm, 'e2', 256, local);
benchmark('vgg_m_face', gpus, useBnorm, 'e2', 192, local);
benchmark('vgg_vd_face', gpus, useBnorm, 'e2', 32, local);
benchmark('resnet50_face', gpus, useBnorm, 'e2', 24, local);

clearDiskSpace('fer');

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
expDefaults('train', model, gpus, useBnorm, lr, bs, local);
expDefaults('test', model, gpus, useBnorm, lr, bs, local);
