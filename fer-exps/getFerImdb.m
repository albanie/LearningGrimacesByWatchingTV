function imdb = getFerImdb(dataDir)
%GETFERIMDB Returns imdb (the image database)
%  IMDB = GETFERIMDB(DATADIR) loads the FER image datbase.  This functions
%  assumes that the raw data `fer2013.csv` has been downloaded and placed
%  in DATADIR.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  rawFaceDims = [48 48 1] ;
  dataPath = fullfile(dataDir, 'fer2013.csv') ;
  csvData = importdata(dataPath) ;

  % parse data
  csvData = csvData(2:end) ; % skip header
  numRows = numel(csvData) ; % skip header
  labels = zeros(1, numRows) ;
  subset = zeros(1, numRows) ;
  imData = zeros(48, 48, 1, numRows, 'single') ;
  parfor ii = 1:numRows
    fprintf('extracting example %d/%d\n', ii, numRows) ;
    tokens = strsplit(csvData{ii}, ',') ;
    labels(ii) = str2double(tokens{1}) + 1 ; % labels need to be one-indexed
    switch tokens{3}
      case 'Training', setIdx = 1 ;
      case 'PublicTest', setIdx = 2 ;
      case 'PrivateTest', setIdx = 3 ;
      otherwise, error('%s not recognised', tokens{3}) ;
    end
    subset(ii) = setIdx ;
    pixels = single(cellfun(@str2double, strsplit(tokens{2}, ' '))) ;
    face = reshape(pixels, rawFaceDims)' ;
    imData(:,:,:,ii) = face ;
  end

  imdb.images.data = imData ;
  imdb.images.labels = labels ;
  imdb.images.set = subset ;
  imdb.meta.sets = {'train', 'val', 'test'} ;
  imdb.meta.classes = {'anger', 'disgust', 'fear', ...
                       'happiness', 'sadness', 'surprise', 'neutral'} ;
