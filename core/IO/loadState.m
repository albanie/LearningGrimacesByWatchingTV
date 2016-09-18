function [dagnet, stats] = loadState(fileName)
% LOADSTATE Loads the network and stats into memory. Returns
% the network as a DagNN object.
%
%   [dagnet, stats] = LOADSTATE(fileName) where fileName specifies
% the path to a .mat file containing these variables.

load(fileName, 'net', 'stats') ;
dagnet = dagnn.DagNN.loadobj(net) ;
