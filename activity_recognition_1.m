%Load data
ios_act_raw = readtable('ios_act.csv');

%Remove unnecassary columns 
ios_act = removevars(ios_act_raw, [1:4,12,13,17,21,25:47,49,50:54]);

head(ios_act,5) % show first 5 rows of table
whos %List variables in workspace

%Massere data før oppdeling

rng(8000,'twister'); %Set the random number generator to a known state

% Define a random partition on a set of data of a specified size
iosactCVP = cvpartition(size(ios_act,1),'holdout',0.2) %holdout = test set
idx = iosactCVP.test;

%seperate to training and test data
dataTrain = ios_act(~idx,:);
dataTest = ios_act(idx,:);

C = table2cell(dataTrain);
C = ctranspose(C); %transpose the columns to rows
dataTrainInput = [C{2,:}];

%prepare data for input



%Define the network architecture
numFeatures = 16;
%The number of hidden units corresponds to the amount of information
%remembered between time steps (the hidden state). The hidden state
%can contain information from all previous time steps, regardless 
%of the sequence length
numHiddenUnits = 100;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%Train the network
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',dataTest, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

%Feiler her
net = trainNetwork(C(:),layers,options);

%Test the network
YPred = classify(net,dataTest,'MiniBatchSize',miniBatchSize);
acc = mean(YPred == YValidation)