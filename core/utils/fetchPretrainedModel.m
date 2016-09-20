function fetchPretrainedModel(modelDir, modelName) 
%FETCHPRETRAINEDMODEL downloads a pretrained network
%   FETCHPRETRAINEDMODEL(modelDir, modelName) downloads a 
%   model named `modelName` and stores it in the directory
%   `modelDir`

switch modelName

    case 'alexnet_imagenet'
        url = 'http://www.vlfeat.org/matconvnet/imagenet-matconvnet-alex.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_m_imagenet'
        url = 'http://www.vlfeat.org/matconvnet/imagenet-vgg-m-1024.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_vd_imagenet'
        url = 'http://www.vlfeat.org/matconvnet/imagenet-matconvnet-vgg-verydeep-16.mat' ;
        fetch(modelDir, url) ;

    case 'resnet50_imagenet'
        url = 'http://www.vlfeat.org/matconvnet/imagenet-resnet-50-dag.mat' ;
        fetch(modelDir, url) ;

    case 'alexnet_face'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/alexnet-face-bn.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_m_face'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/vgg-m-face-bn.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_vd_face'
        url = 'http://www.vlfeat.org/matconvnet/vgg-face.mat' ;
        fetch(modelDir, url) ;

    case 'resnet50_face'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/resnet50-face-bn.mat' ;
        fetch(modelDir, url) ;

    case 'alexnet_face_fer'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/alexnet-face-fer-bn.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_m_face_fer'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/vgg-m-face-fer-bn.mat' ;
        fetch(modelDir, url) ;

    case 'vgg_vd_face_fer'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/vgg-vd-face-fer.mat' ;
        fetch(modelDir, url) ;

    case 'resnet50_face_fer'
        url = 'http://www.robots.ox.ac.uk/~vgg/data/facevalue/resnet50-face-fer-bn.mat' ;
        fetch(modelDir, url) ;
end

% --------------------------------------
function fetch(modelDir, url)
% --------------------------------------

[~, modelFile] = fileparts(url) ;
modelPath = fullfile(modelDir, modelFile) ;
fprintf('downloading %s\n', url) ;
websave(modelPath, url) ;
