stack = params2stack(opt_params, ei);
denom = sqrt(sum(stack{1}.W.^2,2));
X = bsxfun(@rdivide, stack{1}.W, denom);

figure;
for i = 1: size(X,1)
    subplot(16,16,i);
    row = X(i,:);
    imshow(mat2gray(reshape(row, 28,28)));
end