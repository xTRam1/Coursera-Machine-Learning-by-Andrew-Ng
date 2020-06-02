function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

[m, n] = size(X);
X_poly = ones(m, p);

for i = 1:m,
	for j = 1:p,
		X_poly(i,j) = X(i)^j; 
	end;
end;
end
