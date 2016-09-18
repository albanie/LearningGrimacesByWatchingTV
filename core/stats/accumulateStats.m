function stats = accumulateStats(stats_)
% Takes in 'stats_'  a composite object (indexed by GPU number) 
% containing the statistics from the current epoch of training.
% The stats from each GPU are combined into a single struct 'stats'
% which is then returned.

stats = struct() ;

for mode = {'train', 'val'}
  mode = char(mode) ;
  total = 0 ;

  for gpuIdx = 1:numel(stats_)
    stats__ = stats_{gpuIdx} ;
    num__ = stats__.(mode).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(mode))', 'num')
      f = char(f) ;

      if gpuIdx == 1
        stats.(mode).(f) = 0 ;
      end
      stats.(mode).(f) = stats.(mode).(f) + stats__.(mode).(f) * num__ ;

      if gpuIdx == numel(stats_)
        stats.(mode).(f) = stats.(mode).(f) / total ;
      end
    end
  end
  stats.(mode).num = total ;
end
