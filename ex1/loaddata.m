data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X mu sigma] = featureNormalize(X);
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


derivate = zeros(1, length(X));
temp = zeros(1, length(X));


predictions =  X * theta;
errors = (predictions-y);
derivative(j) = sum(sum(errors .* X(:,1)))