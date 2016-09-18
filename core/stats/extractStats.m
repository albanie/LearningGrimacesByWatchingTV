function stats = extractStats(net)
%EXTRACTSTATS returns statistics for the loss layers of the network

sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
    stats.(net.layers(sel(i)).name) = net.layers(sel(i)).block.average ;
end
