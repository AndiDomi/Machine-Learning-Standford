function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




function g = sigmoid(z)
      g=1./(1+e.^(-z));
end


temp = 0;

i = 1;

O0 = theta;

while i <= m,
  h0 = sigmoid(X(i,:)*O0);
  temp = temp + ( ( -y(i)'*log(h0) ) - ( (1-y(i))'*log(1-h0) ) );
  i = i + 1;
end;

j=2;
temp_reg = 0;

while j<= size(theta)(1),
    temp_reg = temp_reg + O0(j);
    j = j+1;
end

J = (temp/m) + (temp_reg*lambda)/(2*m);

temp_01 = 0;
i = 1;
j = 1;
O0 = theta;


   while j <= size(theta)(1),
     while i <= m,
       h0 = sigmoid(X(i,:)*O0);
       temp_01 =  temp_01 + (( h0 - y(i))*X(i)(j));
         i = i + 1;
     end;
      j = j + 1;
   end;

%grad = temp_01 / m;
O0 = theta;
O0(1)=0;
grad = ( (1/m)*( X'*( sigmoid(X*theta)-y))) + (lambda/m)*O0;
  
% working one , but why is working?
%grad = (1/m) * (X' * (sigmoid(X*theta)-y));





% =============================================================

end
