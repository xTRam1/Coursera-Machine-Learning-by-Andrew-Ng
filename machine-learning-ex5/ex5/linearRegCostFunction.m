function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);

J = (1/(2*m)) * sum((X * theta - y).^2, index=1) + (lambda / (2 * m)) * sum(theta(2:size(theta, 1)).^2);
grad = (1/m) * ((X' * (X * theta - y)) + lambda * [0; theta(2:size(theta, 1))]);

end
