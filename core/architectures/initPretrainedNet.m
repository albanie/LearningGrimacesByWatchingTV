function dagnet = initPretrainedNet(opts)

% --------------------------------------------------------------------
%                                            Imagenet-pretrained models
% --------------------------------------------------------------------
if strcmp(opts.pretrainedNet, 'alexnet_imagenet')
    net = load(fullfile(opts.modelDir, 'imagenet-matconvnet-alex.mat'));
end

if strcmp(opts.pretrainedNet, 'vgg_m_imagenet')
    net = load(fullfile(opts.modelDir, 'imagenet-vgg-m.mat'));
end

if strcmp(opts.pretrainedNet, 'vgg_vd_imagenet')
    net = load(fullfile(opts.modelDir, 'imagenet-vgg-verydeep-16.mat'));
end

if strcmp(opts.pretrainedNet, 'resnet50_imagenet')
    net = load(fullfile(opts.modelDir, 'imagenet-resnet-50-dag.mat'));
end

% --------------------------------------------------------------------
%                                           Vgg-face pretrained models
% --------------------------------------------------------------------

if strcmp(opts.pretrainedNet, 'alexnet_face')
    data = load(fullfile(opts.modelDir, 'alexnet-face.mat'));
    net = data.net;
end

% (pretrained by Elliot on vgg-faces with BN)
if strcmp(opts.pretrainedNet, 'vgg_m_face')
    data = load(fullfile(opts.modelDir, 'vgg-m-face.mat'));
    net = data.net;
end

% vgg-vd-16 trained on vgg-faces data without batch normalization
% NOTE: This is "vgg_face"
if strcmp(opts.pretrainedNet, 'vgg_vd_face')
    net = load(fullfile(opts.modelDir, 'vgg-vd-face.mat'));
end

if strcmp(opts.pretrainedNet, 'resnet50_face')
    data = load(fullfile(opts.modelDir, 'resnet-50-face.mat'));
    net = data.net;
end

% --------------------------------------------------------------------
%                                     Vgg-face -> FER pretrained models
% --------------------------------------------------------------------

if strcmp(opts.pretrainedNet, 'alexnet_face_fer')
    data = load(fullfile(opts.modelDir, 'alexnet-face-fer.mat'));
    dagnet = dagnn.DagNN.loadobj(data.net);
    return; % no need for further mods
end

% (pretrained by Elliot on vgg-faces with BN)
if strcmp(opts.pretrainedNet, 'vgg_m_face_fer')
    net = load(fullfile(opts.modelDir, 'vgg-m-face-fer.mat'));
    dagnet = dagnn.DagNN.loadobj(net);
    return; % no need for further mods
end

% vgg-vd-16 trained on vgg-faces data without batch normalization
% NOTE: This is "vgg_face"
if strcmp(opts.pretrainedNet, 'vgg_vd_face_fer')
    net = load(fullfile(opts.modelDir, 'vgg-vd-face-fer.mat'));
    return; % no need for further mods
end

if strcmp(opts.pretrainedNet, 'resnet50_face_fer')
    data = load(fullfile(opts.modelDir, 'resnet-50-face-fer.mat'));
    dagnet = dagnn.DagNN.loadobj(data.net);
    return; % no need for further mods
end

if strcmp(opts.pretrainedNet, 'vgg_m_face_fer_mdond')
    net = load(fullfile(opts.modelDir, 'vgg-m-face-fer.mat'));
    % still needs mods
end


dagnet = opts.modifyForTask(net, opts);
