function dagnet = fixBatchNormDims(dagnet)
%FIXBATCHNORMDIMS fixes the necessary batch norm dimensions
%  FIXBATCHNORMDIMS(dagnet) checks the dimensions of each
%  parameter stored for batch normalization.  If the dims
%  are incorrect, they are fixed

% NOTE: This is primarily to handle oddities in the pretrained
% resnet configuration

for l = 1:numel(dagnet.layers)
    if isa(dagnet.layers(l).block, 'dagnn.BatchNorm')
        paramIdx = dagnet.layers(l).paramIndexes;
        dagnet = modifyParamDims(dagnet, l, paramIdx);
    end
end

function dagnet = modifyParamDims(dagnet, layerIdx, paramIdx)

inputs = dagnet.layers(layerIdx).inputs;
numChannels = dagnet.layers(dagnet.getLayerIndex(inputs)).block.size(4);

for p = paramIdx 
    dagnet.params(p).value = reshape(dagnet.params(p).value, numChannels, []);
end

