function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  
  % initialize objective value and gradient.

  %
  % Compute the logistic regression objective function and gradient 
  % using vectorized code.  (It will be just a few lines of code!)
  % Store the objective function value in 'f', and the gradient in 'g'.
  %
  
  sigmoid = @(x) (1./(1+exp(-x)));
  y_hat = sigmoid(theta'*X);
  f = -(y*log(y_hat)'+(1-y)*log(1 - y_hat)');
  g = X*(y_hat-y)';
  