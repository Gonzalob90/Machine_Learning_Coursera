function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%Create a distance matrix of m x K
distance = zeros(size(X,1),K);

for i = 1:K;
  
  %Create a column vector of the distance from each Training example to that centroid
  diffs = bsxfun(@minus, X, centroids(i,:));
  temp1 = sum(diffs.^2, 2);

  %Store as a column in the distance matrix
  distance(:,i)= temp1; 

end

%return a vector mx1 of the minimum distances
[x, indx] = min(distance');
idx = indx(:);

% =============================================================

end

