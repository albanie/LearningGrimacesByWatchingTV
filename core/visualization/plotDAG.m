function plotDAG(stats, epoch, opts)
%PLOTDAG plots the training and evaluation statistics
% of the DagNN network.

% In test mode, we don't bother plotting
if opts.testMode
    return
end

figure(1) ; clf ;
plots = setdiff(...
    cat(2,...
    fieldnames(stats.train)', ...
    fieldnames(stats.val)'), {'num', 'time'}) ;
for p = plots
    p = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
            tmp = [stats.(f).(p)] ;
            values(end+1,:) = tmp(1,:)' ;
            leg{end+1} = f ;
        end
    end
    subplot(1,numel(plots),find(strcmp(p,plots))) ;
    plot(1:epoch, values','o-') ;
    xlabel('epoch') ;
    title(p) ;
    legend(leg{:}) ;
    grid on ;
end
drawnow ;
print(1, opts.modelFigPath, '-dpdf') ;
