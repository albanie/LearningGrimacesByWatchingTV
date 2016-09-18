function inputs = getBatch(imdb, batch, mode, opts, dagnet)
% returns the inputs and associated labels in the
% batch specified by 'batch', contained in 'imdb'.

input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);

% flip if requested
if strcmp(mode, 'train') && opts.LRflip
    input = augment(input, opts, dagnet);
end

if numel(opts.gpus) > 0
    input = gpuArray(input);
end

inputs = { 'input',input, ...
           'label', label};
end

function input = augment(rawInput, opts, dagnet)
% returns an agumented version of the input to
% help with training.

rawSize = size(rawInput);
input = single(zeros(rawSize));

% flip images with probability 0.5
for idx = 1:size(rawInput, 4)
    img = rawInput(:,:,:,idx);
    
    % random zoom
%     zoom = rand / 10;
%     img = imresize(img, 1 + zoom);
    
    %tfs = [...
    %    .5 0 0 1 1 .5 0 0 1 1 ;
    %    .5 0 1 0 1 .5 0 1 0 1 ;
    %    0 0 0 0 0  1 1 1 1 1] ;
    %
    %[~,transformations] = sort(rand(size(tfs,2), numel(size(rawInput,4)), 1));
    %
    %% crop & flip
    %w = size(img,2) ;
    %h = size(img,1) ;
    %sz = round(min(dagnet.meta.normalization.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
    %dx = randi(w - sz(2) + 1, 1) ;
    %dy = randi(h - sz(1) + 1, 1) ;
    %
    %sx = round(linspace(dx, sz(2)+dx-1, dagnet.meta.normalization.imageSize(2))) ;
    %sy = round(linspace(dy, sz(1)+dy-1, dagnet.meta.normalization.imageSize(1))) ;
    %img = img(sy,sx,:);
    
    if rand > 0.5
        img = fliplr(img);
    end
    input(:,:,:,idx) = img;
end

input = single(input);
end
