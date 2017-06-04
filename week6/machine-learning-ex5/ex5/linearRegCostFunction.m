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



scalar = 1/(2*m);
cumulativeError=sum(((X * theta) - y ).^2);
gradTheta = [ [0] ; theta([2:length(theta)])];;
regulator = (lambda / (2 * m)) * (gradTheta' * gradTheta);
disp(regulator);
J = (scalar * cumulativeError + regulator);

gradScalar = 1/m;
gradError = ((X * theta) - y)'*X;
gradRegulator = (lambda/m).*gradTheta;

grad = ((gradScalar .* gradError)' + gradRegulator);









% =========================================================================

grad = grad(:);

end
