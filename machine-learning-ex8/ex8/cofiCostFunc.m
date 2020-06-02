function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

X = reshape(params(1:(num_movies * num_features), :), num_movies, num_features); 
Theta = reshape(params((num_movies * num_features + 1):end, :), num_users, num_features);
X_Theta_Y = X * Theta' - Y;
J = 0.5 * sum(sum((X_Theta_Y.^2) .* R)) + (0.5 * lambda) * (sum(sum(Theta.^2)) + sum(sum(X.^2)));

[i, j] = find(R==0);
idx = sub2ind(size(Y), i, j);
X_Theta_Y(idx) = 0;

X_grad = X_Theta_Y * Theta + lambda * X; 
Theta_grad = X_Theta_Y' * X + lambda * Theta;
grad = [X_grad(:); Theta_grad(:)];

end
