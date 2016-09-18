function [dagnet, stats] = computeEpoch(state, dagnet, epoch, stats, opts)
% COMPUTEEPOCH distributes the processing work for the given epoch
% according to the number of gpus available.

if opts.numGpus <= 1
    stats = processSequentially(state, dagnet, epoch, stats, opts);
else
    [dagnet, stats] = processInParallel(state, dagnet, epoch, stats, opts);
end
