function learningRate = getLearningRate(epoch, opts)
% GETLEARNINGRATE returns the learning rate for the given epoch. 
% NOTE: If the learningRate for network training has been 
% specifed as an array we perform a validation step to 
% check that a learning rate has been given for every epoch.

% If only a single learning rate is given for training,
% this is used uniformly at every epoch.
if numel(opts.learningRate) == 1
  learningRate = opts.learningRate;
else
  % check learningRate array is valid
  if (numel(opts.learningRate) ~= opts.numEpochs)
      errorId = 'learningRate:size' ;
      msg = 'A learning rate must be specifed for every epoch' ;
      error(errorId, msg) ;
  end
  learningRate = opts.learningRate(epoch) ;
end

end