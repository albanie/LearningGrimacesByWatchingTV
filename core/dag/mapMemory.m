function mmap = mapMemory(dagnet, opts)
% MAPMEMORY creates a memory mapping if more than one gpu 
% is being used.

if numel(opts.gpus) > 1
  mmap = map_gradients(dagnet, opts) ;
else
  mmap = [] ;
end

% -------------------------------------------------------------------------
function mmap = map_gradients(dagnet, opts)
% -------------------------------------------------------------------------
fname = opts.memoryMapFile ;

format = {} ;
for i=1:numel(dagnet.params)
  format(end+1,1:3) = {'single', ...
                       size(dagnet.params(i).value), ...
                       dagnet.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numel(opts.gpus)
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', ...
                  format, 'Repeat', ...
                  numel(opts.gpus), ...
                  'Writable', true) ;
