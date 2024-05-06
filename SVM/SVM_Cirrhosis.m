train_data = real(csvread('..\Data\train.csv',1,1));
train = train_data(:,1:18);
train_label = train_data(:,end);
test_data = real(csvread('..\Data\test.csv',1,1));
test = test_data(:,1:18);
test_label = test_data(:,end);
val_data = real(csvread('..\Data\val.csv',1,1));
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
 
log_loss = loss(mdl,test,test_label,'LossFun','logit');
confusionMat=confusionmat(string(test_label),string(res));
confusionchart(confusionMat);
%% 
accuracy = sum(res == test_label)/length(test_label);
confmat = confusionMat + 1e-10;
precision = @(confmat) diag(confmat)./sum(confmat,2);

recall = @(confmat) diag(confmat)./sum(confmat,1)';

f1Scores = @(confmat) 2*(precision(confmat).*recall(confmat))./(precision(confmat)+recall(confmat));
mean_precision = mean(precision(confmat));
mean_recall = mean(recall(confmat));
mean_f1Score = mean(f1Scores(confmat));