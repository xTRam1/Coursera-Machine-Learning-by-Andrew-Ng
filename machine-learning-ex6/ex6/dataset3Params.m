function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

poss_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
poss_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

best_C_sigma = [0.01, 0.01];
best_acc = 0;

% You need to return the following variables correctly.
for i = 1:length(poss_C),
    for j = 1:length(poss_sigma),
        curr_C = poss_C(i), curr_sigma = poss_sigma(j),
        model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
        curr_acc = length(find((svmPredict(model, Xval) - yval) == 0)) / length(yval),
        visualizeBoundary(Xval, yval, model);
        if curr_acc >= best_acc,
            best_acc = curr_acc;
            best_C_sigma = [curr_C, curr_sigma];
        end; 
    end;
end;

C = best_C_sigma(1), sigma = best_C_sigma(2),

end
