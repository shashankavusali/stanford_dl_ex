function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

stack{1}.W = W1;
stack{1}.b = b1;
stack{2}.W = W2;
stack{2}.b = b2;
numHidden = 1;
numLayers = 3;
hAct = cell(numLayers, 1);
gradStack = cell(numHidden+1, 1);
nsamples = size(data,2);

%% forward prop

hAct{1} = data;
for i = 2 : numLayers
    W = stack{i-1}.W;
    B = stack{i-1}.b;
    Z = bsxfun(@plus,W*hAct{i-1},B);
    hAct{i} = sigmoid(Z);
end

%% Sparsity cost
rho = sparsityParam;
rho_hat = cell(numHidden,1);
for i = 2: numLayers - 1
    rho_hat{i-1} = sum(hAct{i},2)/nsamples;
end

sparsity_cost = beta* sum((rho*log(rho./rho_hat{1})+ (1-rho)*log((1-rho)./(1-rho_hat{1}))),1);

ceCost = 0.5*sum(sum((hAct{numLayers} - data).^2))/nsamples;

%% Backward propagation
delta_l = (hAct{numLayers} - data).*hAct{numLayers}.*(1-hAct{numLayers});

for i = numLayers - 1:-1:1
    gradStack{i}.W = (delta_l * hAct{i}')/nsamples;
    gradStack{i}.b = sum(delta_l,2)/nsamples;
    if i > 1
        delta_l = (bsxfun(@plus,stack{i}.W'*delta_l,beta*(-rho./rho_hat{1} + (1-rho)./(1-rho_hat{1})))).*(hAct{i}.*(1-hAct{i})); 
    end
end

wCost = 0;
for i = 1: numHidden + 1
    wCost = wCost + 0.5 * lambda * sum(stack{i}.W(:).^2);
end

cost = ceCost + wCost + sparsity_cost;

for i = 1 : numHidden + 1
    gradStack{i}.W = gradStack{i}.W + lambda * stack{i}.W; 
end

W1grad = stack{1}.W;
W2grad = stack{2}.W;
b1grad = stack{1}.b;
b2grad = stack{2}.b;
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

