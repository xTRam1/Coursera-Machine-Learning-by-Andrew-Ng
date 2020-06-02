function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

m = size(X, 1);
K = size(centroids, 1);
idx = zeros(m, 1);

for i = 1:m,
    curr_example = X(i,:);
    min_dist = sum((curr_example - centroids(1,:)).^2);
    idx(i) = 1;
    for k = 2:K,
        curr_dist = sum((curr_example - centroids(k,:)).^2);
        if curr_dist < min_dist,
            min_dist = curr_dist;
            idx(i) = k;
        end;
    end;
end; 

end

