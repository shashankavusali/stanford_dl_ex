function [ cost, grad, pred_prob] = autoencoder( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into sum of squares error, weight norm, and prox reg
%        components (ceCost, wCost, pCost)


%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
numLayers = numel(ei.layer_sizes) + 1;
hAct = cell(numLayers, 1);
gradStack = cell(numHidden+1, 1);
nsamples = size(data,2);
%% forward prop

sigmoid = @(x) (1./(1+exp(-x)));

% hAct{1} = data(:,1:5);
hAct{1} = data;
for i = 2 : numLayers
    W = stack{i-1}.W;
    B = stack{i-1}.b;
    Z = bsxfun(@plus,W*hAct{i-1},B);
    hAct{i} = sigmoid(Z);
end

%% compute cost
%%% YOUR CODE HERE %%%
ceCost = 0.5*sum(sum((hAct{numLayers} - data).^2))/nsamples;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

delta_l = (hAct{numLayers} - data).*hAct{numLayers}.*(1-hAct{numLayers});

for i = numLayers - 1:-1:1
    gradStack{i}.W = delta_l * hAct{i}';
    gradStack{i}.b = sum(delta_l,2);
    delta_l = (stack{i}.W'*delta_l).*(hAct{i}.*(1-hAct{i})); 
end

% persistent iter_count;
% if isempty(iter_count)
%     iter_count = 0;
% end
% iter_count = iter_count + 1;
% scatter(iter_count, stack{2}.W(3,3));
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1: numHidden + 1
    wCost = wCost + 0.5 * ei.lambda * sum(stack{i}.W(:).^2);
end

cost = ceCost + wCost;

for i = 1 : numHidden + 1
    gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W; 
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end




