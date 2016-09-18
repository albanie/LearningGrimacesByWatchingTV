function [labels, data] = preprocessSFEW(imgList, opts, dagnet)
% takes in a cell array of img files of the form:
%   /data/raw/SUBSET/EMOTION/IMG_NAME.PNG
% and returns an Nx1 array of emotion labels and a
% h x w x 3 x N data array containing the images.

% Use a reduced image set on my local machine
LOCAL_LIMIT = 1000;
if opts.local
    numImages = min(LOCAL_LIMIT, numel(imgList));
else
    numImages = numel(imgList);
end

labels = cellfun(@extractEmotion, {imgList{1:numImages}});
labels = single(labels);

dataSize = horzcat(dagnet.meta.normalization.imageSize(1:3), numImages);
data = zeros(dataSize);
rawData = cellfun(@(x) extractFace(x, dagnet), {imgList{1:numImages}}, ...
    'UniformOutput', false);

for idx = 1:numImages
    data(:,:,:,idx) = rawData{idx};
end

data = single(data);

% -----------------------------------------
function emotionLabel = extractEmotion(path)
% -----------------------------------------

pathElems = strsplit(path, '/');
emotionKey = pathElems{end - 1};
keys = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'};
values = [1, 2, 3, 4, 5, 6. 7];
emotionMap = containers.Map(keys, values);
emotionLabel = emotionMap(emotionKey);

% -----------------------------------------
function face = extractFace(path, dagnet)
% -----------------------------------------

img = imread(path);
face = normalizeFace(img, dagnet);

% -----------------------------------------
function face = normalizeFace(greyFace, dagnet)
% -----------------------------------------
% Normalizes the network.  This has to be done differently for
% different models

% if we revert to orignal images, they are colour
if size(greyFace, 3) == 3
    greyFace = rgb2gray(greyFace);
end

greyFace = imresize(greyFace, dagnet.meta.normalization.imageSize(1:2));
greyFace = single(greyFace);

% handle average image consisting of color channel averages
sz = size(dagnet.meta.normalization.averageImage);
if sz(1:2) == [3 1]
    face = cat(3, ...
        greyFace - dagnet.meta.normalization.averageImage(1), ...
        greyFace - dagnet.meta.normalization.averageImage(2), ...
        greyFace - dagnet.meta.normalization.averageImage(3));

% handle grayscale average image
elseif sz(3) == 1
    face = greyFace - dagnet.meta.normalization.averageImage;

% handle full pixel average image
else
    face = cat(3, ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,1), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,2), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,3));
end
