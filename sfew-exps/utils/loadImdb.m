function imdb = loadImdb(opts, dagnet, mode)
% Returns imdb (the image database). If the imdb file has already been
% created previously, load into memory. Otherwise, build it from scratch.

% In 'train' mode, the training and validation imdb is loaded.
% In 'test' mode, the test imdb is loaded.

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    mkdir(opts.expDir);
    mkdir(fileparts(opts.imdbPath));
    if strcmp(mode, 'train')
        imdb = getFacesImdb(opts, dagnet);
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
    else % mode == 'test'
        imdb = getTestFacesImdb(opts, dagnet);
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
    end
end


%imdb.images.set = 3 - imdb.images.set ;

end

% --------------------------------------------------------------------
function imdb = getFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

trainImgs = get_pngs_in_subdirs(fullfile(opts.dataDir, 'raw', 'Train'));
[trainLabels, trainData] = preprocessSFEW(trainImgs, opts, dagnet);

valImgs = get_pngs_in_subdirs(fullfile(opts.dataDir, 'raw', 'Val'));
[valLabels, valData] = preprocessSFEW(valImgs, opts, dagnet);

% get the set labels for the training data
trainSet = ones(1, size(trainLabels,2));
train_label = 1;
[trainSet(:)] = deal(train_label);

% get the set labels for the validation data
valSet = 2 * ones(1, size(valLabels,2));
valLabel = 2;
[valSet(:)] = deal(valLabel);

% create the imdb training/validation structure expected by matconvnet
train_val_data = cat(4, trainData, valData);
train_val_labels = cat(2, trainLabels, valLabels);
train_val_set = cat(2, trainSet, valSet);

imdb.images.data = train_val_data;
imdb.images.labels = train_val_labels;
imdb.images.set = train_val_set;
imdb.meta.sets = {'train', 'val'};
imdb.meta.classes = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'};

end

% --------------------------------------------------------------------
function imdb = getTestFacesImdb(opts, dagnet)
% --------------------------------------------------------------------

testImgs = get_pngs_in_subdirs(fullfile(opts.dataDir, 'raw', 'Test'));
[testFaces, names] = preprocessTestSFEW(testImgs, opts, dagnet);

imdb.images.data = testFaces;
imdb.images.names = names;
imdb.meta.sets = {'test'};
imdb.meta.classes = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'};
end
