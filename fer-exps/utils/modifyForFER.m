function dagnet = modifyForFER( net, opts )
%MODIFYFORFER prepares a pretrained network for FER
%   MODIFYFORFER(net, varargin) performs the appropriate
%   modifications to the last few layers of the network to
%   make it suitable for classification on the FER dataset
%   and converts to a DagNN
%
% Args:
%   net (struct): the pretrained network (SimpleNN or DagNN)
%   opts (struct): the options used to configure the network
%

% Returns:
%   DagNN: the modified network
%

% set defaults
opts.numOutputs = 7;

% prepare dagNN from input network, which can be either
% a SimpleNN or a DagNN
if ~isfield(net, 'params') % SimpleNN
    dagnet = prepareFromSimpleNN(net, opts);
else % DagNN
    dagnet = prepareFromDagNN(net, opts);
end

% configure the loss layers
dagnet = configureForClassification(dagnet, opts);

% modify the network to us Batch Normalization if needed
if opts.useBnorm
    dagnet = insertBNLayers(dagnet, opts);
end