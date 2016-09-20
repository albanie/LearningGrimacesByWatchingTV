function experimentScheduler(gpus)
%EXPERIMENTSCHEDULER runs the given list of experiments
%   on the gpus provided
%
%   NOTES: 
%   The batch size used to train the vgg_vd_imagenet model (10)
%   is smaller than the size (32) used originally in the paper to take 
%   advantage of the newer pre-trained model (performance should be 
%   comparable or slightly better)
%   
%   The original vgg-vd-16-face model was trained without batch 
%   normalization so we use the same approach here when using the matconvnet
%   release model. In practice, we found that pretraining vgg-vd-16 from 
%   scratch with batch normalization on the VGG faces dataset achieved the 
%   same results on FER

startup ;

if nargin == 0
    gpus = [];
end

useBnorm = true;
local = false;

% imagenet-pretrained models
%benchmark('alexnet_imagenet', gpus, 256, useBnorm, 'e2', local) ;
%benchmark('vgg_m_imagenet', gpus, 192, useBnorm, 'e2', local) ;
%benchmark('vgg_vd_imagenet', gpus, 10, useBnorm, 'e2', local) ; 
%benchmark('resnet50_imagenet', gpus, 24, useBnorm, 'e2', local) ;

% VGGFace-pretrained models 
%benchmark('alexnet_face', gpus, 256, useBnorm, 'e2', local) ;
%benchmark('vgg_m_face', gpus, 192, useBnorm, 'e2', local) ;
%benchmark('vgg_vd_face', gpus, 32, false, 'e2', local) ; 
%benchmark('resnet50_face', gpus,  24, useBnorm, 'e2', local) ;

% rather than training models, we can also download an evaluate 
% the fully trained models
%benchmarkPretrained('alexnet_face_fer', gpus, 256) ;
%benchmarkPretrained('vgg_m_face_fer', gpus, 192) ;
%benchmarkPretrained('vgg_vd_face_fer', gpus, 32) ;
benchmarkPretrained('resnet50_face_fer', gpus, 32) ;

clearDiskSpace('fer');

% -------------------------------------------------------------------------
function benchmark(model, gpus, bs, useBnorm, lr, local)
% -------------------------------------------------------------------------
% BENCHMARK trains and tests a model
%   BENCHMARK(model, gpus, useBnorm, lr, bs, local) trains and tests 
%   the given model using the provided parameters
%   
%   'model' (string): the name of the model
%   'useBnorm' (boolean): train the network with Batch Normalization
%   'gpus' (Array): a list of gpu devices to be used
%   'bs' (scalar): batch size
%   'lr' (scalar/Array): learning rate schedule
%   'local' (boolean): run the experiment "locally" with only a small
%       subset of the training data

expDefaults('train', model, gpus,  bs, useBnorm, lr,local);
expDefaults('test', model, gpus,  bs, useBnorm, lr, local);

% -------------------------------------------------------------------------
function benchmarkPretrained(model, gpus, bs)
% -------------------------------------------------------------------------
% BENCHMARKPRETRAINED evaluates a pretrained model
%   BENCHMARKPRETRAINED(model, gpus, useBnorm, bs, local) evaluates 
%   the given model using the provided parameters
%   
%   'model' (string): the name of the pretrained model
%   'gpus' (Array): a list of gpu devices to be used
%   'bs' (scalar): batch size

expDefaults('test-deployed', model, gpus, bs) ;
