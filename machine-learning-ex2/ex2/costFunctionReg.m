function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);

J = (1 / m) * (-1 * ((y' * log(sigmoid(X * theta))) + ((1 - y)' * log(1 - sigmoid(X * theta)))) + (lambda / 2) * (sum(theta(2:length(theta)).^2, index=1)));

grad = (1 / m) * (X' * (sigmoid(X * theta) - y));
grad(2:length(grad)) += (lambda / m) * theta(2:length(theta));

end
