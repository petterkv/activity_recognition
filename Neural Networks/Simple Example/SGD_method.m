%The generalized delta rule using the SGD

%Weights, training data (input) and correct_Output as input, Weight as output
function Weight = SGD_method(Weight, input, correct_Output)
alpha = 0.9

N = 4; %Iterate 4 times

for k = 1:N
    transposed_Input = input(k, :)';
    d = correct_Output (k);
weighted_Sum = Weight * transposed_Input; %Calculate the weighted sum
output = Sigmoid(weighted_Sum); %Pass wighted_Sum to the Sigmoid function

error = d - output;
delta = output*(1-output)*error;

dWeight = alpha*delta*transposed_Input;

Weight(1) = Weight(1) + dWeight(1);
Weight(2) = Weight(2) + dWeight(2);
Weight(3) = Weight(3) + dWeight(3);

end
end
