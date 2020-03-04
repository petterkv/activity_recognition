% The activation function

%Define the functions output(y) and input (x). Sigmoid is the name of the
%function
function y = Sigmoid(x)
y = 1/(1+exp(-x)); %The implementation of the function
end
