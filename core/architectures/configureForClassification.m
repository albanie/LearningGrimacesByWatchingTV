function dagnet = configureForClassification(dagnet, opts)
%CONFIGUREFORCLASSIFICATION configures the network to train as a classifer
%  CONFIGUREFORCLASSIFICATION(dagnet, opts) adds a softmaxlog loss 
%   and classerror loss on top of the fully connected output predictions
%   of the network to perform classification.  
%
%   A fine tuning learning rate is set on each of the network parameters.
%   Appropriate meta information is also added for the emotion recognition
%   task

% set the learning rates for fine tuning
paramIdx = dagnet.getParamIndex([dagnet.layers(1:end-2).params]);
[dagnet.params(paramIdx).learningRate] = deal(opts.fineTuningRate);

% Add softmaxlog loss layer (for training)
layer = dagnn.Loss('loss', 'softmaxlog');
inputs = {'prediction','label'};
output = 'objective';
dagnet.addLayer('loss', layer, inputs, output);

% Add class error
layer = dagnn.Loss('loss', 'classerror');
inputs = {'prediction','label'};
output = 'classerror';
dagnet.addLayer('classerror', layer, inputs, output) ;

dagnet.rebuild()

% modify the meta attributes of the net
dagnet.meta.classes.name = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'};
dagnet.meta.classes.description = {'anger', 'disgust', 'fear', ...
    'happiness', 'sadness', 'surprise', 'neutral'};
