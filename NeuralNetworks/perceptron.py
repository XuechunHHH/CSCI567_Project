import numpy as np
import random
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from algorithm import Algorithm
from analysis import get_trials_summary, plot_trials

class Perceptron(Algorithm):
    def __init__(self, classes):
        self.classes = classes

    def trainGD(self, X, y, num_epochs):
        self.weights = np.zeros((self.classes, len(X[0])))
        
        mistakes = []

        for _ in range(num_epochs):
            mistake = 0
            for index, row in enumerate(X):
                yhat = np.argmax(row @ self.weights.T)
                if yhat != y[index]:
                    self.weights[y[index]] = self.weights[y[index]] + row
                    self.weights[yhat] = self.weights[yhat] - row
                    mistake += 1

            mistakes.append(mistake)

        return mistakes
    
    def trainSGD(self, X, y, num_epochs):
        self.weights = np.zeros((self.classes, len(X[0])))
        iters = len(X) * num_epochs

        mistakes = 0

        for _ in range(iters):
            index = random.randint(0, len(X) - 1)
            row = X[index]
            yhat = np.argmax(row @ self.weights.T)
            if yhat != y[index]:
                self.weights[y[index]] = self.weights[y[index]] + row
                self.weights[yhat] = self.weights[yhat] - row
                mistakes += 1

        return mistakes
    
    def forward_pass(self, X, y):
        results = X @ self.weights.T
        return np.argmax(results, axis=1), None

def main():
    training_data, training_label, validation_data, validation_label, testing_data, testing_label = preprocess_data()

    output_size = 3
    
    num_epochs_set = [10, 20, 30, 40, 50]
    num_trials = 100

    avg = []

    for num_epochs in num_epochs_set:
        model = Perceptron(classes=output_size)
        _ = model.trainGD(training_data, training_label, num_epochs)
        train_trial = model.get_acc(training_data, training_label)
        test_trial = model.get_acc(validation_data, validation_label)
        avg.append([train_trial, test_trial])

        model.get_confusion_matrix(validation_data, validation_label, output_size)
        if num_epochs == 10:
            print("Metrics for Gradient Descent: ")
            model.get_metrics(testing_data, testing_label)

    plot_trials(avg, None, num_epochs_set)

    avg = []
    std = []

    for num_epochs in num_epochs_set:
        train_trials = []
        test_trials = []
        for _ in range(num_trials):
            model = Perceptron(classes=3)
            _ = model.trainSGD(training_data, training_label, num_epochs)
            train_trials.append(model.get_acc(training_data, training_label))
            test_trials.append(model.get_acc(validation_data, validation_label))

        # model.get_confusion_matrix(validation_data, validation_label, output_size)
        if num_epochs == 10:
            print("Metrics for Stochastic Gradient Descent: ")
            model.get_metrics(testing_data, testing_label)

        summary = get_trials_summary(train_trials, test_trials)
        avg.append([summary[0], summary[1]])
        std.append([summary[2], summary[3]])

    plot_trials(avg, std, num_epochs_set)

if __name__ == "__main__":
    main()