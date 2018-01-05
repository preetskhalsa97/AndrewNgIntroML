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

% Part 1

% Add ones to the X data matrix
X = [ones(m, 1) X];

%making a matrix of hOfTheta to keep a track for all the training examples

hOfThetaCumulative = zeros(num_labels,m);
yCumulative = zeros(num_labels,m);
z2Cumulative = zeros(size(Theta1,1),m);
a2Cumulative = zeros(size(Theta1,1)+1,m);
% For every i, calculating hOfTheta(Xi)

for i=1:m,
  %for the second (hidden) layer
  Xi = X(i,:)'; % column vector
  z2 = Theta1*Xi;
  z2Cumulative(:,i) = z2;
  a2 = sigmoid(z2);
  %for the third (output) layer
  a2 = [1 ; a2];
  z3 = Theta2*a2;
  hOfTheta = sigmoid(z3);
  a2Cumulative(:,i) = a2;
  hOfThetaCumulative(:,i) = hOfTheta;
  
  % for each of the output label k
  
  Yi = zeros(num_labels,1);
  Yi(y(i)) = 1;
  
  yCumulative(:,i) = Yi;
  
  for k=1:num_labels,
    J = J + Yi(k)*log(hOfTheta(k)) + (1-Yi(k))*log(1-hOfTheta(k));   
  end;
  
end;

J = J*(-1*1/m);

%implementing regularization
Jreg = 0;
%for the second layer, i.e., components of Theta1
%going over all the rows
for i=1:size(Theta1,1),
  %going over all columns except the first one
  for j=2:size(Theta1,2),
    Jreg = Jreg + Theta1(i,j)^2;
  end;
end;


%for the third layer, i.e., components of Theta2
%going over all the rows
for i=1:size(Theta2,1),
  %going over all columns except the first one
  for j=2:size(Theta2,2),
    Jreg = Jreg + Theta2(i,j)^2;
  end;
end;

Jreg = Jreg*lambda/(2*m);

J = J + Jreg;

%Part 2

Del3 = zeros(num_labels,1);

Delta1 = zeros(size((Theta2'*hOfThetaCumulative(:,1))(2:end)*a2Cumulative(:,1)'));
Delta2 = zeros(size(hOfThetaCumulative(:,1)*hOfThetaCumulative(:,1)'));

for i=1:m,
  Del3 = hOfThetaCumulative(:,size(hOfThetaCumulative,2)) - yCumulative(:,i);
  a = sigmoid(z2Cumulative(:,i));
  a  = [1;a];
  Del2 = Theta2'*Del3.*[sigmoidGradient(a)];
  Del2 = Del2(2:end);
  
  Delta1 = Delta1 + Del2*a2Cumulative(:,i)';
  
  Delta2 = Delta2 + Del3*[hOfThetaCumulative(:,i)]';
  
  
end;

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

end

