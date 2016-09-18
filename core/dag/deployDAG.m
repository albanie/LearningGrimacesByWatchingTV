function net = deployDAG(net)
% DEPLOYDAG  Deploy a DagNN network by merging the batch
% normalization layers into the preceding convolutions.

dagMergeBatchNorm(net) ;
dagRemoveLayersOfType(net, 'dagnn.BatchNorm') ;

% Switch to use MatConvNet default memory limit for CuDNN (512 MB)
for name = dagFindLayersOfType(net, 'dagnn.Conv')
    l = net.getLayerIndex(char(name)) ;
    net.layers(l).block.opts = removeCuDNNMemoryLimit(net.layers(l).block.opts) ;
end

% -------------------------------------------------------------------------
function opts = removeCuDNNMemoryLimit(opts)
% -------------------------------------------------------------------------
remove = false(1, numel(opts)) ;
for i = 1:numel(opts)
    if isstr(opts{i}) && strcmp(lower(opts{i}), 'CudnnWorkspaceLimit')
        remove([i i+1]) = true ;
    end
end
opts = opts(~remove) ;

% -------------------------------------------------------------------------
function layers = dagFindLayersWithOutput(net, outVarName)
% -------------------------------------------------------------------------
layers = {} ;
for l = 1:numel(net.layers)
    if any(strcmp(net.layers(l).outputs, outVarName))
        layers{1,end+1} = net.layers(l).name ;
    end
end

% -------------------------------------------------------------------------
function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = [] ;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
    end
end

% -------------------------------------------------------------------------
function dagRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, type) ;
for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end

% -------------------------------------------------------------------------
function dagMergeBatchNorm(net)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, 'dagnn.BatchNorm') ;
for name = names
    name = char(name) ;
    layer = net.layers(net.getLayerIndex(name)) ;
    
    % merge into previous conv layer
    player = dagFindLayersWithOutput(net, layer.inputs{1}) ;
    player = net.layers(net.getLayerIndex(player)) ;
    if ~isa(player.block, 'dagnn.Conv')
        error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
    end
    
    filters = net.getParamIndex(player.params{1}) ;
    biases = net.getParamIndex(player.params{2}) ;
    multipliers = net.getParamIndex(layer.params{1}) ;
    offsets = net.getParamIndex(layer.params{2}) ;
    moments = net.getParamIndex(layer.params{3}) ;
    
    [filtersValue, biasesValue] = mergeBatchNorm(...
        net.params(filters).value, ...
        net.params(biases).value, ...
        net.params(multipliers).value, ...
        net.params(offsets).value, ...
        net.params(moments).value) ;
    
    net.params(filters).value = filtersValue ;
    net.params(biases).value = biasesValue ;
end

% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% wk / sqrt(sigmak^2 + eps)
% bk - wk muk / sqrt(sigmak^2 + eps)
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
biases(:) = biases(:) + b(:) ;
sz = size(filters) ;
numFilters = sz(4) ;
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
