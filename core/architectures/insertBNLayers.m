function dagnet = insertBNLayers(dagnet, opts)
%INSERTBNLAYERS adds batch normalization to a network
%  INSERTBNLAYERS(dagnet) inserts batch normalization
%  layers directly after each convolutional layer in the
%  the DagNN network

% First check if the network already contains batch norm
% layers - if so, make sure that the shapes are correct
for l = 1:numel(dagnet.layers)
    if isa(dagnet.layers(l).block, 'dagnn.BatchNorm')
        if findstr('resnet50_imagenet', opts.pretrainedNet)
            dagnet = fixBatchNormDims(dagnet);
        end
        return;
    end
end

% Loop over the network and insert batch norm layers
layerOrder = dagnet.getLayerExecutionOrder();
for l = layerOrder
    if isa(dagnet.layers(l).block, 'dagnn.Conv') && ...
            ~strcmp(dagnet.layers(l).outputs, 'prediction')
        dagnet = addBatchNorm(dagnet, l);
    end
end

dagnet.rebuild()


function dagnet = addBatchNorm(dagnet, layerIndex)
%ADDBATCHNORM adds batch normalization to a network
%  INSERTBNLAYERS(dagnet, opts) inserts batch normalization
%  layers directly after each convolutional layer in the
%  the DagNN network

% the inputs and outputs are chosen to insert the BN at 
% the correct location in the network
inputs = dagnet.layers(layerIndex).outputs;

% find the number of channels produced by the previous layer
numChannels = dagnet.layers(layerIndex).block.size(4);

outputs = sprintf('xbn%d',layerIndex);

% Define the name and parameters for the new layer
name = sprintf('bn%d', layerIndex);

block = dagnn.BatchNorm();
paramNames = {sprintf('%sm', name) ...
              sprintf('%sb', name) ...
              sprintf('%sx', name) };
          
% add new layer to the network          
dagnet.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    paramNames) ;


% set mu (gain parameter)
mIdx = dagnet.getParamIndex(paramNames{1});
dagnet.params(mIdx).value = ones(numChannels, 1, 'single');
dagnet.params(mIdx).learningRate = 2;
dagnet.params(mIdx).weightDecay = 0;

% set beta (bias parameter)
bIdx = dagnet.getParamIndex(paramNames{2});
dagnet.params(bIdx).value = zeros(numChannels, 1, 'single');
dagnet.params(bIdx).learningRate = 1;
dagnet.params(bIdx).weightDecay = 0;

% set moments parameter
xIdx = dagnet.getParamIndex(paramNames{3});
dagnet.params(xIdx).value = zeros(numChannels, 2, 'single');
dagnet.params(xIdx).learningRate = 0.05;
dagnet.params(xIdx).weightDecay = 0;

% check that the next layer is a ReLU as expected
assert(isa(dagnet.layers(layerIndex + 1).block, 'dagnn.ReLU'));

% modify the next layer to take the new inputs
dagnet.layers(layerIndex + 1).inputs = {outputs};
