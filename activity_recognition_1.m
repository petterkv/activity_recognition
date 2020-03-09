%% Load data
ios_act_raw = readtable('ios_act.csv');
%Remove unnecassary columns 
ios_act = removevars(ios_act_raw, [1:4,12,13,17,21,25:47,49,50:54]);
head(ios_act,5) % show first 5 rows of table

%% Prepare training and validation set
whos %List variables in workspace


%Massere data før oppdeling

rng(8000,'twister'); %Set the random number generator to a known state

% Define a random partition on a set of data of a specified size
iosactCVP = cvpartition(size(ios_act,1),'holdout',0.2) %holdout = test set
idx = iosactCVP.test;

%seperate to training and test data
dataTrain = ios_act(~idx,:);
dataTest = ios_act(idx,:);
%% Transpose data
%C = table2cell(dataTrain); %Convert to cell array
%C = ctranspose(C); %transpose the columns to rows
%Derive all the predictors for all the obseravtions. The predictors are
%represented one one line after one orher 16 16 16 16 16
%dataTrainInputX = [C{1:16,:}]; 
%dataTrainInputX = tonndata([dataTrain{1:16,:}],true,false);
%celldata = transpose(table2cell([dataTrain(:,1:16)]));
%cell = table2cell(dataTrain(:,1:16));
%trans = transpose(dataTrain(:,1:16));

%td = transpose(tonndata(dataTrain{1:16,:}, true, false));
[predictors, wasMatrix] = tonndata(dataTrain{1:10,1:16},false,false);
dependentcategorical = ({transpose(categorical(dataTrain{1:10,17}))});
%[dependent, wasMatrix2] = tonndata(dependentcategorical,false,false);
%[dependent, wasMatrix2] = tonndata(dataTrain{1:10,17},false,false);

%dataTrainInputX = transpose(tonndata(dataTrain{1:10,1:16}, false, false));


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

%% 
net = trainNetwork(predictors, {dependentcategorical}, layers, options);

%Test the network
YPred = classify(net,dataTest,'MiniBatchSize',miniBatchSize);
acc = mean(YPred == YValidation)