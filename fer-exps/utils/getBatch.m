function inputs = getBatch(imdb, batch, mode, opts, dagnet)
% returns the inputs and associated labels in the
% batch specified by 'batch', contained in 'imdb'.

rawInput = imdb.images.data(:,:,:,batch) ;
label = imdb.images.labels(1, batch) ;
outSize = [ dagnet.meta.normalization.imageSize(1:3) numel(batch) ] ;
input = single(zeros(outSize)) ;

for idx = 1:numel(batch)

    img = rawInput(:,:,:,idx) ;

    % normalize face image 
    img = normalizeFace(img, dagnet) ;
    
    % flip images with probability 0.5 if requested
    if strcmp(mode, 'train') && opts.LRflip
      if rand > 0.5
        img = fliplr(img) ;
     end
    end

    input(:,:,:,idx) = img ;
end

if numel(opts.gpus) > 0
    input = gpuArray(input) ;
end

inputs = { 'input',input, ...
           'label', label} ;

% --------------------------------------------------------------------
function face = normalizeFace(greyFace, dagnet)
% --------------------------------------------------------------------
% Normalizes the network (done differently for different models)

greyFace = imresize(greyFace, dagnet.meta.normalization.imageSize(1:2)) ;

% handle average image consisting of color channel averages
sz = size(dagnet.meta.normalization.averageImage) ;
if sz(1:2) == [3 1]
    face = cat(3, ...
        greyFace - dagnet.meta.normalization.averageImage(1), ...
        greyFace - dagnet.meta.normalization.averageImage(2), ...
        greyFace - dagnet.meta.normalization.averageImage(3)) ;

% handle grayscale average image
elseif sz(3) == 1
    face = greyFace - dagnet.meta.normalization.averageImage ;

% handle full pixel average image
else
    face = cat(3, ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,1), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,2), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,3)) ;
end
