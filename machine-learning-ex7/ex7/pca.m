function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

m = size(X, 1);
sigma = (1/m) * (X' * X);
[U,S,V] = svd(sigma);

end
