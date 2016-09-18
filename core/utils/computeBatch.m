function batch = computeBatch(subsetIdx, subBatchIdx, subset, opts)
% COMPUTEBATCH computes the indices to be used in the next batch of 
% data.
batchStart = subsetIdx + (labindex-1) + (subBatchIdx-1) * numlabs ;
batchEnd = min(subsetIdx + opts.batchSize-1, numel(subset));
batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd);
end