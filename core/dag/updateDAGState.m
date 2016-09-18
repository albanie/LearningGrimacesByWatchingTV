function state = updateDAGState(state, epoch, opts)
% UPDATEDAGSTATE returns an updated state structure for
% the DAG in which the training data has been shuffled.

state.epoch = epoch;
state.val = opts.val;
state.learningRate = getLearningRate(epoch, opts);

% Shuffle the training data indices
state.train = opts.train(randperm(numel(opts.train)));
