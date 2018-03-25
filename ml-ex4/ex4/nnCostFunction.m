function [J grad] = nnCostFunction(nn_params, ...
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
Y = zeros(m, num_labels);
for i=1:m
  Y(i, y(i))= 1;
end

a1 = [ones(m,1) X];
z2u = a1*Theta1';
z2 = sigmoid(z2u);
[x,~]=size(z2);
a2 = [ones(x,1) z2];
a3 = sigmoid(a2*Theta2');
H = a3;

J = (1/m)*sum(sum(-Y.*log(H) + (Y-1).*log(1-H)));

J = J + (lambda/(2*m))*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ); 

D1=0;
D2=0;

d3 = a3 - Y;
%z2 = [ones(m,1) z2];

%size(z2u)
%size(Theta2(:,2:end))
%size(d3)

d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2u);
%d2 = d2(:,2:end);

%a2
%a3
%d2
%d3
%z2u
%sigmoidGradient(z2)

D1 = D1 + d2'*a1;
D2 = D2 + d3'*a2;

%D1 %- 25*401
%D2 %- 10*26

%size(Theta1(:,2:end))
%size(Theta2(:,2:end))

Theta1_grad = (1/m).*D1;
Theta2_grad = (1/m).*D2; 

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m).*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m).*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
