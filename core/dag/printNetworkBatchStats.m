function printNetworkBatchStats(state, stats, opts, subsetIdx, subset, start, mode)
%PRINTNETWORKBATCHSTATS prints out summary statistics for the current batch
% under processing

time = toc(start) ;
stats.time = toc(start) ;

batchNumber = fix(subsetIdx/opts.batchSize)+1 ;
totalNumberOfBatches = ceil(numel(subset)/opts.batchSize) ;
batchSpeed = stats.num/stats.time * max(numel(opts.gpus), 1) ;

fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    state.epoch, ...
    batchNumber, ...
    totalNumberOfBatches, ...
    batchSpeed) ;

for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:%.3f', f, stats.(f)) ;
end
fprintf('\n') ;
