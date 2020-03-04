%Load data
%[XTrain,YTrain] = japaneseVowelsTrainData;
%[XValidation,YValidation] = japaneseVowelsTestData;
ios_act_raw = readtable('ios_act.csv');
head(ios_act_raw,5) % show first 5 rows of table

%Massere data før oppdeling

whos 
rng(8000,'twister'); %Set the random number generator to a known state
holdoutCVP = cvpartition(size(ios_act_raw,1),'holdout',0.2)
idx = holdoutCVP.test;

%seperate to training and test data
dataTrain = ios_act_raw(~idx,:);
dataTest = ios_act_raw(idx,:);

%XTrain(1:5)

%Define the network architecture
numFeatures = 57;
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
net = trainNetwork(dataTrain,layers,options);

%Test the network
YPred = classify(net,dataTest,'MiniBatchSize',miniBatchSize);
acc = mean(YPred == YValidation)