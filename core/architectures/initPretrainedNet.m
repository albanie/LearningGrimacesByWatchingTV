function dagnet = initPretrainedNet(opts)

% --------------------------------------------------------------------
%                                            Imagenet-pretrained models
% --------------------------------------------------------------------

if strcmp(opts.pretrainedNet, 'alexnet_imagenet')
    modelFile = 'imagenet-matconvnet-alex.mat' ;
end

if strcmp(opts.pretrainedNet, 'vgg_m_imagenet')
    modelFile = 'imagenet-vgg-m.mat' ;
end

if strcmp(opts.pretrainedNet, 'vgg_vd_imagenet')
    modelFile = 'imagenet-vgg-verydeep-16.mat' ;
end

if strcmp(opts.pretrainedNet, 'resnet50_imagenet')
    modelFile = 'imagenet-resnet-50-dag.mat' ;
end

% --------------------------------------------------------------------
%                                           Vgg-face pretrained models
% --------------------------------------------------------------------

if strcmp(opts.pretrainedNet, 'alexnet_face')
    modelFile = 'alexnet-face-bn.mat' ;
end

if strcmp(opts.pretrainedNet, 'vgg_m_face')
    modelFile = 'vgg-m-face-bn.mat' ;
end

if strcmp(opts.pretrainedNet, 'vgg_vd_face')
    modelFile = 'vgg-face.mat' ;
end

if strcmp(opts.pretrainedNet, 'resnet50_face')
    modelFile = 'resnet50-face-bn.mat' ;
end


% --------------------------------------------------------------------
%                                     Vgg-face -> FER pretrained models
% --------------------------------------------------------------------

if strcmp(opts.pretrainedNet, 'alexnet_face_fer')
    modelFile = 'alexnet-face-fer-bn.mat' ;
    net = loadModel(opts.modelDir, modelFile, opts) ;
    dagnet = dagnn.DagNN.loadobj(net);
    return; % no need for further mods
end

if strcmp(opts.pretrainedNet, 'vgg_m_face_fer')
    modelFile = 'vgg-m-face-fer-bn.mat' ;
    net = loadModel(opts.modelDir, modelFile, opts) ;
    dagnet = dagnn.DagNN.loadobj(net);
    return; % no need for further mods
end

if strcmp(opts.pretrainedNet, 'vgg_vd_face_fer')
    modelFile = 'vgg-vd-face-fer.mat' ;
    net = loadModel(opts.modelDir, modelFile, opts) ;
    dagnet = dagnn.DagNN.loadobj(net);
    return; % no need for further mods
end

if strcmp(opts.pretrainedNet, 'resnet50_face_fer')
    modelFile = 'resnet50-face-fer-bn.mat' ;
    net = loadModel(opts.modelDir, modelFile, opts) ;
    dagnet = dagnn.DagNN.loadobj(net);
    return; % no need for further mods
end

net = loadModel(opts.modelDir, modelFile, opts) ;
dagnet = opts.modifyForTask(net, opts);

% --------------------------------------------------------------------
function net = loadModel(modelDir, modelFile, opts) 
% --------------------------------------------------------------------

modelPath = fullfile(modelDir, modelFile) ;
if ~exist(modelPath)
    fetchPretrainedModel(modelDir, opts.pretrainedNet) ;
end

net = load(modelPath);
