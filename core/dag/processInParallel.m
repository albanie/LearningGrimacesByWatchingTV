function [dagnet, stats] = processInParallel(state, dagnet, epoch, stats, opts)
% PROCESSINPARALLEL distributes the work of processing an epoch across
% the available gpus.

    % convert dagnet to vanilla matlab struct
    savedNet = dagnet.saveobj() ;
    spmd
      dagnet_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = processEpoch(state, dagnet_, opts, 'train') ;
      stats_.val = processEpoch(state, dagnet_, opts, 'val') ;
      if labindex == 1, savedNet_ = dagnet_.saveobj() ; end
    end
    dagnet = dagnn.DagNN.loadobj(savedNet_{1}) ; 
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
end
