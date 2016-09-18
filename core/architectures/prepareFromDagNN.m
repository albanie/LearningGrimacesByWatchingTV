function dagnet = prepareFromDagNN(net, opts)
%PREPAREFROMDAGNN prepares a DagNN structure for classification
%  PREPARESFROMIMPLENN(dagnet, opts) prepares a DagNN 
%  network for classification by removing any old loss layers 
%  and ensuring that the final fully connected "prediction" 
%  layer has the correct dimensions.

% load stored network into memory 
dagnet = dagnn.DagNN.loadobj(net);

% remove previous loss layers / softmax layers
for l = numel(dagnet.layers): -1:1
    if isa(dagnet.layers(l).block, 'dagnn.Loss') 
        dagnet.removeLayer(dagnet.layers(l).name);
    elseif isa(dagnet.layers(l).block, 'dagnn.SoftMax') 
        dagnet.removeLayer(dagnet.layers(l).name);
    end
end

% modify last fully connected layer for multi-way classification
layerOrder = dagnet.getLayerExecutionOrder();
finalLayer = dagnet.layers(layerOrder(end));
numChannels = finalLayer.block.size(3);
finalLayer.block.size = [1 1 numChannels opts.numOutputs];

% Initialize the params of the new prediction layer
rng('default'); rng(0);
fScale = 1/100;
filters = fScale * randn(1, 1, numChannels, opts.numOutputs, 'single');
biases = zeros(opts.numOutputs, 1, 'single');

% Handle possible naming conventions for parameters:
% Filter params can be called: 
% <layername>f, <layername>_filter, <layername>_f
filterIdx = dagnet.getParamIndex(sprintf('%sf', finalLayer.name));
if isnan(filterIdx)
    filterIdx = dagnet.getParamIndex(sprintf('%s_filter', finalLayer.name));
end
if isnan(filterIdx)
    filterIdx = dagnet.getParamIndex(sprintf('%s_f', finalLayer.name));
end

% Bias params can be called: <layername>b, <layername>_bias, <layername>_b
biasIdx = dagnet.getParamIndex(sprintf('%sb', finalLayer.name));
if isnan(biasIdx)
    biasIdx = dagnet.getParamIndex(sprintf('%s_bias', finalLayer.name));
end
if isnan(biasIdx)
    biasIdx = dagnet.getParamIndex(sprintf('%s_b', finalLayer.name));
end

dagnet.params(filterIdx).value = filters;
dagnet.params(biasIdx).value = biases;

% Rename input variable to 'input' be consistent with other networks
firstLayer = dagnet.layers(layerOrder(1));
if ~strcmp(firstLayer.inputs, 'input')
    dagnet.renameVar(firstLayer.inputs, 'input');
end

% rename the output of the last fully connected layer to "prediction"
predictionVar = dagnet.layers(dagnet.getLayerIndex(finalLayer.name)).outputs;
dagnet.renameVar(predictionVar, 'prediction');
