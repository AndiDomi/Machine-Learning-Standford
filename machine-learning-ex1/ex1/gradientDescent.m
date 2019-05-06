function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

disp(theta)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
temp_O0 = 0;
temp_O1 = 0;
i = 1;
 O0 = theta(1);
 O1 = theta(2);
while i <= m,
  h0 = O0 + O1*X(i,2);
  temp_O0 = temp_O0 + (h0 - y(i));
  temp_O1 = temp_O1 + ((h0 - y(i))*X(i,2));
  i = i + 1;
end;

theta(1) = theta(1) - alpha*(temp_O0)/m;
theta(2) = theta(2) - alpha*(temp_O1)/m;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
