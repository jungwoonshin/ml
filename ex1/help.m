data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
theta = zeros(2, 1); % initialize fitting parameters

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;


m = size(X,1); % row length of the matrix ; number of training example
predictions =  X * theta;
sqrErrors = (predictions-y).^2;
