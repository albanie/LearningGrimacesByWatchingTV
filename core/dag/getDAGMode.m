function evaluateMode = getDAGMode(opts)
% GETDADGMODE returns true if the DAG is currently in 
% validation mode and false if the DAG is in training mode.

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
