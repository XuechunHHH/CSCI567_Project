train_data = real(csvread('train.csv',1,1));
train = train_data(:,1:18);
train_label = train_data(:,end);
test_data = real(csvread('test.csv',1,1));
test = test_data(:,1:18);
test_label = test_data(:,end);
val_data = real(csvread('val.csv',1,1));
val = val_data(:,1:18);
val_label = val_data(:,end);
temp_svm = templateSVM('Standardize',false,'KernelFunction','linear');
%   mdl = fitcecoc(train, train_label,'Coding','onevsone','Learners',temp_svm, ...
%         'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
%        'HyperparameterOptimizationOptions',struct('Kfold',5));
  mdl = fitcecoc([train;val], [train_label;val_label],'Learners',temp_svm, ...
        'OptimizeHyperparameters',{'BoxConstraint','Coding','KernelFunction'}, ...
       'HyperparameterOptimizationOptions',struct('Kfold',5));
% mdl = fitcecoc(train, train_label,'Coding','onevsone','Learners',temp_svm);
res = mdl.predict(test);
 
confusionMat=confusionmat(string(test_label),string(res));
confusionchart(confusionMat);
%% 
accuracy = sum(res == test_label)/length(test_label);
confusionMat = confusionMat + 1e-10;
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);

recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';

f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
mean_precision = mean(precision(confusionMat));
mean_recall = mean(recall(confusionMat));
mean_f1Score = mean(f1Scores(confusionMat));