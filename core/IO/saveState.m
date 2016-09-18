function saveState(fileName, dagnet, stats)
% SAVESTATE saves the DagNN object `dagnet` as a vanilla matlab 
% structure `net`, together with the stats collected so far in 
% the file specified by `fileName`.

net = dagnet.saveobj();
save(fileName, 'net', 'stats');
