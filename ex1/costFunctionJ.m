function J = costFunctionJ(X, y, theta)

m = size(X,1); % row length of the matrix ; number of training example
predictions = X * theta;
sqrErrors = (predictions-y).^2;

J = 1/(2*m) * sum(sqrErrors);

