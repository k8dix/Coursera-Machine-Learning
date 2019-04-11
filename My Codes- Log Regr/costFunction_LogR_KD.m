function [J, grad] = costFunction_LogR_KD(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X * theta);
unregCost = (1/m) * (((-y') * (log(h))) - (((1-y)') * (log(1-h))));
theta(1) = 0;
thetaSqr = (theta' * theta);
lambda = 0;
scale = (lambda / (2 * m));
regCost = thetaSqr * scale;
J = unregCost + regCost;

unregGrad = (1/m) * ((X') * (h - y));
theta(1) = 0;
regGrad = (theta * (lambda / m));
grad = unregGrad + regGrad;

%grad = gradient descent







% =============================================================

end
