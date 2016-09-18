function dagnet = prepareFromSimpleNN(net, opts)
%PREPAREFROMSIMPLENN prepares a SimpleNN for classification
%  PREPARESFROMIMPLENN(dagnet, opts) prepares a SimpleNN 
%  network for classification by removing any old loss layers 
%  and ensuring that the final fully connected "prediction" 
%  layer has the correct dimensions.
%
%  NOTE: Once prepared, the network is converted to a DagNN 
%  for further processing

% remove previous softmax layers
for l = 1:numel(net.layers)
    if ismember(net.layers{l}.type, {'softmax', 'softmaxloss'})
        net.layers(l) = [];
    end
end

% modify last fully connected layer for multi-way classification
rng('default'); rng(0);
fScale = 1/100;
numChannels = size(net.layers{end}.weights{1}, 3);
filters = fScale * randn(1, 1, numChannels, opts.numOutputs, 'single');
biases = zeros(1, opts.numOutputs, 'single');
modifiedWeights = { filters, biases };
net.layers{end}.weights = modifiedWeights;
lastLayerName = net.layers{end}.name;

% convert to dagNN
dagnet = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);

% rename the output of the last fully connected layer to "prediction"
predictionVar = dagnet.layers(dagnet.getLayerIndex(lastLayerName)).outputs;
dagnet.renameVar(predictionVar, 'prediction');
