clear all

rootDir = 'E:\我的论文\高时间分辨率拉曼细菌识别\code\data';
classes = {'CJ','EC','LM','SA','ST'};
numPerFile = 100; 
imgSize = [20, 2000, 1];
z = 1:5:100;

allData = [];
allLabels = [];

for classIdx = 1:length(classes)
    className = classes{classIdx};
    classFolder = fullfile(rootDir, className);
    files = dir(fullfile(classFolder, '*.mat'));
    
    for i = 1:length(files)
        f = load(fullfile(classFolder, files(i).name)); 
        data = f.spectra;
        data = data(z,:);

        allData(:, :, end+1) = data; 
        allLabels(end+1) = classIdx; 

    end
end

allData(:,:,1) = [];

labelCategories = categorical(allLabels, 1:5, classes);

numTotal = size(allData, 3);
rng(1);
indices = randperm(numTotal);
numTrain = round(0.6 * numTotal);

XTrain = allData(:, :, indices(1:numTrain));
YTrain = labelCategories(indices(1:numTrain));

XTest = allData(:, :, indices(numTrain+1:end));
YTest = labelCategories(indices(numTrain+1:end));

XTrain = reshape(XTrain, 20, 2000, 1, numTrain);
XTest =  reshape(XTest, 20, 2000, 1, numTotal - numTrain);

dsTrain = augmentedImageDatastore(imgSize, XTrain, YTrain);
dsTest = augmentedImageDatastore(imgSize, XTest, YTest);

clear allData XTrain XTest

inputSize = imgSize;
numClasses = 5;

layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')

    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')

    fullyConnectedLayer(128, 'Name', 'fc3')
    reluLayer('Name', 'relu_fc3')

    fullyConnectedLayer(numClasses, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');



net = trainNetwork(dsTrain, layers, options);

YPred = classify(net, dsTest);
accuracy = sum(YPred == YTest) / numel(YTest);

figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for 5-Class Raman CNN');