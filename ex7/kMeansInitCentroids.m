function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% Initialize the centroids to be random examples

% Randomly reorder the indices of examples, X is size 16384 x 3 (m x n)
% randix will reorder the indices and will return a 1x16384 vector
% A [1 2 3], randidx will be [2,1,3]
randidx = randperm(size(X, 1));


% Take the first K examples as centroids
% Then take the first K elements, that meeans the first 3 values, in this case will get 3x3 matrix of centroids (since X has n=3 features)
centroids = X(randidx(1:K), :);


% =============================================================

end

