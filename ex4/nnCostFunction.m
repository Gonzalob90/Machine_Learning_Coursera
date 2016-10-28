function [J_reg grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Convert the y vector into y matrix
y_matrix = eye(num_labels)(y,:);

% Start Forward propagation

a1 = [ones(m,1) X]; % a1 is a (m x n+1) matrix

z2 = a1 * Theta1'; %Theta1 is a (h x n+1) matrix, z2 is a (m x h) matrix

a2 = [ones(m,1) sigmoid(z2)];

z3 = a2 * Theta2';

a3 = sigmoid(z3); %a3 is a (m x k) matrix

% Calculate the cost function 

temp1 = sum(sum((-y_matrix) .* log(a3)));

temp2 = sum(sum((1-y_matrix) .* log(1-a3))); 

J = (1/m * (temp1 - temp2));

%Remove 0s from Theta matrixs

Theta1 = Theta1(:,2:end);
Theta2 = Theta2(:,2:end);

reg_1 = sum(sum(Theta1 .* Theta1));

reg_2 = sum(sum(Theta2 .* Theta2));

J_reg = J + (lambda/(2*m) * (reg_1 + reg_2));

%Start Backpropation

d3 = a3 - y_matrix; %d3 is a (m x k) matrix

d2 = (d3 * Theta2) .* sigmoidGradient(z2); % Theta 2 is (k x h) matrix, d2 is (m x h)

D1 = d2' * a1; % D1 is a (h x n+1) matrix

D2 = d3' * a2; % D2 is a (k x h+1) matrix

Theta1_grad = 1/m .* D1;
Theta2_grad = 1/m .* D2;

% Gradient Regularization

reg_grad1 = (lambda/m) * [zeros(size(Theta1),1) Theta1];

reg_grad2 = (lambda/m) * [zeros(size(Theta2),1) Theta2];


Theta1_grad = Theta1_grad + reg_grad1;
Theta2_grad = Theta2_grad + reg_grad2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
