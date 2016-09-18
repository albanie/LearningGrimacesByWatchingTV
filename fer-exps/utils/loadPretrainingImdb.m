function imdb = loadPretrainingImdb(opts, dagnet, mode)
% Returns imdb (the image database). If the imdb file has already been
% created previously, load into memory. Otherwise, build it from scratch.

% ALL TEZ DATAA USED FOR PRETRAINING 
% NOTE:
% The subset labels have been switch to treat all data as training

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    mkdir(opts.expDir);
    mkdir(fileparts(opts.imdbPath));
    if strcmp(mode, 'train')
        imdb = getFacesImdb(opts, dagnet);
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
    else % mode == 'test'  
        % NO TEST SET 
    end
end

end

% --------------------------------------------------------------------
function imdb = getFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

trainCsv= csvread(strcat(opts.dataDir, 'raw/train.csv'));
trainLabels = trainCsv(:,1);
trainFaces = trainCsv(:, 2:end);

% Use a reduced training set on my local machine
if opts.local
    numTrainImages = 120;
else
    numTrainImages = numel(trainLabels);
end

% We transform the labels so that they are 1-indexed, rather than
% 0-indexed.
trainLabels = transformLabels(trainLabels(1:numTrainImages)');
trainData = zeros(horzcat(dagnet.meta.normalization.imageSize(1:3), numTrainImages));

for idx = 1:numTrainImages
    greyFace = reshape(trainFaces(idx,:), [48 48])';
    face = normalizeFace(greyFace, dagnet);
    trainData(:,:,:,idx) = face;
end

% Cast all training data to single data type
trainData = single(trainData);
trainLabels = single(trainLabels);

valCsv = csvread(strcat(opts.dataDir, 'raw/val.csv'));
valLabels = valCsv(:, 1);
valFaces = valCsv(:, 2:end);

% Use a reduced validation set on my local machine
if opts.local
    numValImages = 110;
else
    numValImages = numel(valLabels);
end

valLabels = transformLabels(valLabels(1:numValImages)');
valData = zeros(horzcat(dagnet.meta.normalization.imageSize(1:3), numValImages));

for idx = 1:numValImages
    greyFace = reshape(valFaces(idx,:), [48 48])';
    face = normalizeFace(greyFace, dagnet);
    valData(:,:,:,idx) = face;
end

% Cast all validation data to single data type
valData = single(valData);
valLabels = single(valLabels);



testCsv = csvread(strcat(opts.dataDir, 'raw/test.csv'));
testLabels = testCsv(:,1);
testFaces = testCsv(:, 2:end);

% Use a reduced test set on my local machine
if opts.local
    numTestImages = 100;
else
    numTestImages = numel(testLabels);
end


testLabels = transformLabels(testLabels(1:numTestImages)');
testData = single(zeros(horzcat(dagnet.meta.normalization.imageSize(1:3), ...
    numTestImages)));

for idx = 1:numTestImages
    greyFace = single(reshape(testFaces(idx,:), [48 48])');
    face = normalizeFace(greyFace, dagnet);
    testData(:,:,:,idx) = face;
end


% get the set labels for the test data
testSet = 1 * ones(1, size(testLabels,2));
testLabel = 1;
[testSet(:)] = deal(testLabel);

% get the set labels for the training data
trainSet = ones(1, size(trainLabels,2));
trainMarker= 1;
[trainSet(:)] = deal(trainMarker);

% get the set labels for the validation data
valSet = 1 * ones(1, size(valLabels,2));
valMarker = 1;
[valSet(:)] = deal(valMarker);

% add an extremely small val set
miniValData = valData(:,:,:,1:110);
miniValLabels = valLabels(1:110);
miniValSet = 2 * ones(1,110);

% create the imdb training/validation structure expected by matconvnet
train_val_test_data = cat(4, trainData, valData, testData, miniValData);
train_val_test_labels = cat(2, trainLabels, valLabels, testLabels, miniValLabels);
train_val_test_set = cat(2, trainSet, valSet, testSet, miniValSet);

imdb.images.data = train_val_test_data ;
imdb.images.labels = train_val_test_labels;
imdb.images.set = train_val_test_set;
imdb.meta.sets = {'train'};
imdb.meta.classes = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'};

end

function labels = transformLabels(labels)
% we need to add 1 to all emotion labels to ensure that
% 0 (which represents a null label in matconvnet) isn't
% used.

labels = labels + 1;
end

function face = normalizeFace(greyFace, dagnet)
% Normalizes the network.  This has to be done differently for
% different models

greyFace = imresize(greyFace, dagnet.meta.normalization.imageSize(1:2));

if size(dagnet.meta.normalization.averageImage, 3) == 1
    face = greyFace - dagnet.meta.normalization.averageImage;
else
    face = cat(3, ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,1), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,2), ...
        greyFace - dagnet.meta.normalization.averageImage(:,:,3));
end
end