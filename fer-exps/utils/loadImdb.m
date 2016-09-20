function imdb = loadImdb(opts, dagnet, mode)
%LOADIMDB Returns imdb (the image database). If the imdb file has already been
% created previously, load into memory. Otherwise, build it from scratch.

% In 'train' mode, the training and validation imdb is loaded.
% In 'test' mode, the test imdb is loaded.

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    mkdir(opts.expDir) ;
    mkdir(fileparts(opts.imdbPath)) ;
    if strcmp(mode, 'train')
        imdb = getFacesImdb(opts, dagnet) ;
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
    else % mode == 'test'
        imdb = getTestFacesImdb(opts, dagnet) ;
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
    end
end


% --------------------------------------------------------------------
function imdb = getFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

rawFaceDims = [48 48 1] ;
trainCsv = csvread(fullfile(opts.dataDir, 'raw', 'train.csv')) ;
trainLabels = trainCsv(:,1) ;
trainFaces = trainCsv(:, 2:end) ;

% Use a reduced training set on my local machine
if opts.local
    numTrainImages = 120 ;
else
    numTrainImages = numel(trainLabels) ;
end

% We transform the labels so that they are 1-indexed, rather than
% 0-indexed.
trainLabels = transformLabels(trainLabels(1:numTrainImages)') ;
trainData = zeros(horzcat(rawFaceDims, numTrainImages)) ;

for idx = 1:numTrainImages
    face = reshape(trainFaces(idx,:), rawFaceDims)' ;
    trainData(:,:,:,idx) = face ;
end

% Cast all training data to single data type
trainData = single(trainData) ;
trainLabels = single(trainLabels) ;

valCsv = csvread(fullfile(opts.dataDir, 'raw', 'val.csv')) ;
valLabels = valCsv(:, 1) ;
valFaces = valCsv(:, 2:end) ;

% Use a reduced validation set on my local machine
if opts.local
    numValImages = 110 ;
else
    numValImages = numel(valLabels) ;
end

valLabels = transformLabels(valLabels(1:numValImages)') ;
valData = zeros(horzcat(rawFaceDims, numValImages)) ;

for idx = 1:numValImages
    face = reshape(valFaces(idx,:), rawFaceDims)' ;
    valData(:,:,:,idx) = face ;
end

% Cast all validation data to single data type
valData = single(valData) ;
valLabels = single(valLabels) ;

% get the set labels for the training data
trainSet = ones(1, size(trainLabels,2)) ;
trainMarker= 1 ;
[trainSet(:)] = deal(trainMarker) ;

% get the set labels for the validation data
valSet = 2 * ones(1, size(valLabels,2)) ;
valMarker = 2 ;
[valSet(:)] = deal(valMarker) ;

% create the imdb training/validation structure expected by matconvnet
train_val_data = cat(4, trainData, valData) ;
train_val_labels = cat(2, trainLabels, valLabels) ;
train_val_set = cat(2, trainSet, valSet) ;

imdb.images.data = train_val_data ;
imdb.images.labels = train_val_labels ;
imdb.images.set = train_val_set ;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'} ;

% --------------------------------------------------------------------
function imdb_test = getTestFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

rawFaceDims = [48 48 1] ;
testCsv = csvread(fullfile(opts.dataDir, 'raw', 'test.csv')) ;
testLabels = testCsv(:,1) ;
testFaces = testCsv(:, 2:end) ;

% Use a reduced test set on my local machine
if opts.local
    numTestImages = 100 ;
else
    numTestImages = numel(testLabels) ;
end

testLabels = transformLabels(testLabels(1:numTestImages)') ;
testData = single(zeros(horzcat(rawFaceDims, numTestImages))) ;

for idx = 1:numTestImages
    face = single(reshape(testFaces(idx,:), rawFaceDims)') ;
    testData(:,:,:,idx) = face ;
end


% get the set labels for the test data
testSet = 3 * ones(1, size(testLabels,2)) ;
testLabel = 3 ;
[testSet(:)] = deal(testLabel) ;

% To help with issues with average precision, it helps to shuffle
% the data
testIdx = 1:numel(testLabels) ;
shuffledIdx = testIdx(randperm(numel(testLabels))) ;
shuffledLabels = testLabels(shuffledIdx) ;
shuffledData = testData(:,:,:,shuffledIdx) ;

imdb_test.images.data = shuffledData ;
imdb_test.images.labels = shuffledLabels ;
imdb_test.images.set = testSet ;
imdb_test.meta.sets = {'test'}  ;
imdb.meta.classes = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'} ;

% --------------------------------------------------------------------
function labels = transformLabels(labels)
% --------------------------------------------------------------------
% we need to add 1 to all emotion labels to ensure that
% 0 (which represents a null label in matconvnet) isn't
% used.

labels = labels + 1 ;
