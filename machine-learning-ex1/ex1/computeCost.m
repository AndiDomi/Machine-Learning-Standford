function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
temp = 0;
i = 1;
O0 = theta(1);
O1 = theta(2);

while i <= m,

  h0 = O0 + O1*X(i,2);
  temp = temp + (h0 - y(i)).^2;
  i = i + 1;
end;

J = (temp)/(2*m);

end
