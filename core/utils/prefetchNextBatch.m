function prefetchNextBatch(batchStart, subsetIdx, subBatchIdx, subset, state, opts)
% PREFETCHNEXTBATCH saves time by fetching the next
% batch.

if subBatchIdx == opts.numSubBatches
    batchStart = subsetIdx + (labindex-1) + opts.batchSize ;
    batchEnd = min(subsetIdx + 2*opts.batchSize - 1, numel(subset)) ;
else
    batchStart = batchStart + numlabs ;
end
nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
state.getBatch(state.imdb, nextBatch, opts) ;