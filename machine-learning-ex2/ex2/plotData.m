function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

ones = find(y == 1); zeros = find(y == 0);

plot(X(ones, 1), X(ones, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(zeros, 1), X(zeros, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

end
