function [data, names] = preprocessTestSFEW(imgList, opts, dagnet)
% takes in a cell array of img files of the form:
%   /data/raw/SUBSET/EMOTION/IMG_NAME.PNG
% and returns an h x w x 3 x N data array containing 
% the images.

% Use a reduced image set on my local machine 
LOCAL_LIMIT = 1000;
if opts.local
   numImages = min(LOCAL_LIMIT, numel(imgList)); 
else
    numImages = numel(imgList);
end

names = cellfun(@extractFileNames, imgList, 'UniformOutput', false);

dataSize = horzcat(dagnet.meta.normalization.imageSize(1:3), numImages);
data = zeros(dataSize);
rawData = cellfun(@(x) extractFace(x, dagnet), {imgList{1:numImages}}, ...
    'UniformOutput', false);

for idx = 1:numImages
    data(:,:,:,idx) = rawData{idx};
end


data = single(data);

end

% -----------------------------------------
function face = extractFace(path, dagnet)
% -----------------------------------------

img = imread(path);

% First rescale to ensure that the smaller dimension matches the 
% network default requirement.
resizedImg = imresize(img, dagnet.meta.normalization.imageSize(1:2));

resizedImg = single(resizedImg);

% Normalize each colour channel
normalizedImg = single(zeros(dagnet.meta.normalization.imageSize));
for channel = 1:3
    normalizedImg(:,:,channel) = resizedImg - ...
                dagnet.meta.normalization.averageImage(:,:, channel);
end

face = normalizedImg;
end

% -----------------------------------------
function name = extractFileNames(path)
% -----------------------------------------
[~, name ~] = fileparts(path);
end


