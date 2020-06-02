function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = [computeCost(X, y, theta)]

for i=1:num_iters,
    theta = theta - (alpha / m) .* (X'* (X * theta - y));
    J_history = [J_history, computeCost(X, y, theta)];
end

end
