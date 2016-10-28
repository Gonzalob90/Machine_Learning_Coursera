function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Calculate the cost

h = X*theta; % X is 12x2, theta is 2x1, h is 12x1
temp1 = h - y; %h is 12x1, y es 12x1, temp1 es 12x1
J = 1/(2*m) * temp1' * temp1; 

theta(1)=0;

J = J + (lambda/(2*m) * (theta' * theta));

grad = 1/m * (X')*(h-y); %h-y is 12x1, X' is 2x12, grad is 2x1

grad = grad + (lambda/m*theta);

% =========================================================================

grad = grad(:);

end
