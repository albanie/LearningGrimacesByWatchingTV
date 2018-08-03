function dag = insertBNLayers(dag)
%INSERTBNLAYERS adds batch normalization to a network
%  INSERTBNLAYERS(dag) inserts batch normalization
%  layers directly after each convolutional layer in the
%  the DagNN network

  % First check if the network already contains batch norm
  % layers - if so, make sure that the shapes are correct
  for l = 1:numel(dag.layers)
    if isa(dag.layers(l).block, 'dagnn.BatchNorm')
      return ;
    end
  end
  % Loop over the network and insert batch norm layers
  layerOrder = dag.getLayerExecutionOrder() ;
  for l = layerOrder
    if isa(dag.layers(l).block, 'dagnn.Conv') && ...
           ~strcmp(dag.layers(l).outputs, 'prediction')
       dag = addBatchNorm(dag, l) ;
    end
  end
  dag.rebuild()

% ----------------------------------------------------------------------------
function dag = addBatchNorm(dag, layerIndex)
% ----------------------------------------------------------------------------
%ADDBATCHNORM adds batch normalization to a network
%  INSERTBNLAYERS(dag, layerIndex) inserts batch normalization
%  layers directly after each convolutional layer in the
%  the DagNN network

  % the inputs and outputs are chosen to insert the BN at
  % the correct location in the network
  inputs = dag.layers(layerIndex).outputs ;

  % find the number of channels produced by the previous layer
  numChannels = dag.layers(layerIndex).block.size(4) ;

  outputs = sprintf('xbn%d',layerIndex) ;

  % Define the name and parameters for the new layer
  name = sprintf('bn%d', layerIndex) ;

  block = dagnn.BatchNorm() ;
  paramNames = {sprintf('%sm', name) ...
                sprintf('%sb', name) ...
                sprintf('%sx', name) } ;

  % add new layer to the network
  dag.addLayer(name, block, inputs, outputs, paramNames)  ;

  % set mu (gain parameter)
  mIdx = dag.getParamIndex(paramNames{1}) ;
  dag.params(mIdx).value = ones(numChannels, 1, 'single') ;
  dag.params(mIdx).learningRate = 2 ;
  dag.params(mIdx).weightDecay = 0 ;

  % set beta (bias parameter)
  bIdx = dag.getParamIndex(paramNames{2}) ;
  dag.params(bIdx).value = zeros(numChannels, 1, 'single') ;
  dag.params(bIdx).learningRate = 1 ;
  dag.params(bIdx).weightDecay = 0 ;

  % set moments parameter
  xIdx = dag.getParamIndex(paramNames{3}) ;
  dag.params(xIdx).value = zeros(numChannels, 2, 'single') ;
  dag.params(xIdx).learningRate = 0.05 ;
  dag.params(xIdx).weightDecay = 0 ;

  % check that the next layer is a ReLU as expected
  assert(isa(dag.layers(layerIndex + 1).block, 'dagnn.ReLU')) ;

  % modify the next layer to take the new inputs
  dag.layers(layerIndex + 1).inputs = {outputs} ;
