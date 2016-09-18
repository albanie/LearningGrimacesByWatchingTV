function plotROC(net, imdb_test, score)
%PLOTROC Takes in a structure 'net' which contains
% the trained parameters, imdb_test, the test set
% and score, a struct containing the confidence of 
% predictions on the test set.
% Produces a ROC figure.

% score = classifier_score(net, imdb_test);
confidences = [score.confidence];
[sortedConfidences, sortIdx] = sort(confidences, 'descend');

% Initialize state for the T. Fawcett ROC algorithm
% (Algorithm 1 "Efficient method for generating ROC points"
% described in "An Introduction to ROC analysis, 2005"
% We define "flat" to be positive, and "steep" to be negative
% in the language used to describe the algorithm.
sortedLabels = imdb_test.images.labels(sortIdx);

ROC_points = [];
TP = 0; % true positives
FP = 0; % false positives
P = sum(sortedLabels(:) == 1); % total positive samples
N = sum(sortedLabels(:) == 2); % total negative samples
confidence_prev = -Inf;

for i=1:numel(sortedLabels)
    if sortedConfidences(i) ~= confidence_prev
        ROC_point = [ FP/N, TP/P ]';
        ROC_points = horzcat(ROC_points, ROC_point);
        confidence_prev = sortedConfidences(i);
    end
    if sortedLabels(i) == 1
        TP = TP + 1;
    else
        FP = FP + 1;
    end
end

% The final point to be added should be (1,1):
ROC_point = [ FP/N, TP/P ]';
ROC_points = horzcat(ROC_points, ROC_point);    

% display figure
figure
X = ROC_points(1,:);
Y = ROC_points(2,:);
a=5;
c = linspace(1,10, length(X));
scatter(X, Y, a, c);
hold on;
% plot straight line (equivalent to random guessing)
X = linspace(0,1,10);
Y = linspace(0,1,10);
plot(X,Y);

title('ROC curves')
legend('Alexnet', 'Random guessing', 'Location', 'southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
save('AlexNet-Binary4/ROC_points', 'ROC_points');
