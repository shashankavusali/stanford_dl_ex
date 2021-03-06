function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % Compute the objective function by looping over the dataset and summing
  % up the objective values for each example.  Store the result in 'f'.
  
  sigmoid = @(x) (1/(1+exp(-x)));
  
  for i = 1:m
      f = f + (y(i)*log(sigmoid(theta'*X(:,i)))+ (1-y(i))*log(1-sigmoid(theta'*X(:,i))));
  end
  f = -f;
  %
  % Compute the gradient of the objective by looping over the dataset and summing
  % up the gradients (df/dtheta) for each example. Store the result in 'g'.
  
  for i = 1:m
      g = g + X(:,i)*(sigmoid(theta'*X(:,i)) - y(i));
  end
