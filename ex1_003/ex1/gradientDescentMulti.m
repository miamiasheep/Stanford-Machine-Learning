function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

J_history = zeros(num_iters, 1);
fprintf('gridient Descent multi');
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    temp = [0;0;0];
    temp(1)=0;
    temp(2)=0;
    temp(3)=0;
    for i= 1:m
        %temp = temp+(X(i,:)*theta-y(i))*X(i,:)';
        temp(1) = temp(1)+(X(i,:)*theta-y(i))*X(i,1);
        temp(2) = temp(2)+(X(i,:)*theta-y(i))*X(i,2);
        temp(3) = temp(3)+(X(i,:)*theta-y(i))*X(i,3);
    end
        theta(1) = theta(1) - (alpha/m)*temp(1);
        theta(2) = theta(2) - (alpha/m)*temp(2);
        theta(3) = theta(3) - (alpha/m)*temp(3);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
