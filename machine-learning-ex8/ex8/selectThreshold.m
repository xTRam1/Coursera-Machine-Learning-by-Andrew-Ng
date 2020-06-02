function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
stepsize = (max(pval) - min(pval)) / 1000;
for e = min(pval):stepsize:max(pval),
    binary_pred = pval < e;
    tp = sum((binary_pred == 1) & (yval == 1));
    fp = sum((binary_pred == 1) & (yval == 0));
    fn = sum((binary_pred == 0) & (yval == 1));
    rec = tp / (tp + fn);
    prec = tp / (tp + fp);
    F1 = (2 * prec * rec) / (prec + rec);
    if F1 > bestF1,
        bestF1 = F1;
        bestEpsilon = e;
    end;
end
