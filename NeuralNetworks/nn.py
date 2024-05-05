import numpy as np
from preprocess import preprocess_data
from algorithm import Algorithm
from analysis import get_trials_summary, plot_trials
import seaborn as sn
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_size, output_size, lr=0.005):
        self.weights = np.random.normal(0, 0.1, size=(input_size, output_size))
        self.biases = np.random.normal(0, 0.1, size=(1, output_size))
        self.lr = lr

    # forward pass
    def forward(self, X):
        output = X @ self.weights + self.biases
        return output

    # back propogation
    def backprop(self, X, grad):
        gradient_X = grad @ self.weights.T
        self.update(X, grad)
        return gradient_X
    
    # update weights and biases during backprop
    def update(self, X, grad):
        self.weights -= X.T @ grad * self.lr
        self.biases = np.sum(grad, axis=0) * self.lr

class Activation:
    def __init__(self, func='ReLu'):
        if func == 'ReLu':
            self.func = ReLu()
        elif func == 'LeakyReLu':
            self.func = LeakyReLu()
        elif func == 'Sigmoid':
            self.func = Sigmoid()
        else:
            raise Exception('Invalid activation function.')
        
    def forward(self, X):
        return self.func.forward(X)
    
    def backprop(self, X, grad):
        return self.func.backprop(X, grad)
    
class ReLu:
    # forward pass
    def forward(self, X):
        return np.maximum(X, 0)

    # back propogation
    def backprop(self, X, grad):
        pz = (X >= 0).astype(int)
        return pz * grad

class LeakyReLu:
    # forward pass
    def forward(self, X):
        pass

    # back propogation
    def backprop(self):
        pass

class Sigmoid:
    # forward pass
    def forward(self, X):
        pass

    # back propogation
    def backprop(self):
        pass

class Softmax:
    # forward pass
    def forward(self, X, y):
        X = X.astype('float64')
        X_exp = np.exp(X - np.max(X, axis=1).reshape(len(X), 1))
        self.p = X_exp / np.sum(X_exp, axis=1).reshape(len(X), 1)

        pred = np.argmax(self.p, axis=1)

        if y is None:
            return pred, None
        return pred, np.sum(-np.log(self.p[range(len(y)), y])) / len(y)

    # back propogation
    def backprop(self, y):
        y_expand = np.zeros_like(self.p)
        y_expand[range(len(y)), y.flatten()] = 1
        return self.p - y_expand
    
class NeuralNetwork(Algorithm):
    def __init__(self, input_size, l1, l2, output_size, lr=0.005):
        self.layer1 = Layer(input_size, l1, lr)
        self.relu1 = Activation(func='ReLu')
        self.layer2 = Layer(l1, l2, lr)
        self.relu2 = Activation(func='ReLu')
        self.layer3 = Layer(l2, output_size, lr)
        self.softmax = Softmax()

    def forward_pass(self, x, y):
        self.a1 = self.layer1.forward(x)
        self.o1 = self.relu1.forward(self.a1)
        self.a2 = self.layer2.forward(self.o1)
        self.o2 = self.relu2.forward(self.a2)
        self.a3 = self.layer3.forward(self.o2)
        self.loss = self.softmax.forward(self.a3, y)
        return self.loss

    def backward_pass(self, x, y):
        grad_a3 = self.softmax.backprop(y)
        grad_o2 = self.layer3.backprop(self.o2, grad_a3)
        grad_a2 = self.relu2.backprop(self.a2, grad_o2)
        grad_o1 = self.layer2.backprop(self.o1, grad_a2)
        grad_a1 = self.relu1.backprop(self.a1, grad_o1)
        self.layer1.backprop(x, grad_a1)

def main():
    training_data, training_label, validation_data, validation_label, testing_data, testing_label = preprocess_data()

    training_label = training_label.flatten()
    validation_label = validation_label.flatten()
    testing_label = testing_label.flatten()

    input_size = len(training_data[0])

    num_trials = 30

    l1 = 10
    l2 = 6
    output_size = 3
    lr = 0.005

    num_epochs_set = [10, 20, 30, 40, 50]
    batch_size = 3

    rows = len(training_data)

    avg = []
    std = []

    for num_epochs in num_epochs_set:
        print("Number of Epochs: {}".format(num_epochs))

        train_trials = []
        test_trials = []
        for _ in range(num_trials):
            # initialize neural network
            model = NeuralNetwork(input_size, l1, l2, output_size, lr)

            # start training
            for _ in range(num_epochs):
                for i in range(0, rows, batch_size):
                    end = min(rows, i + batch_size)
                    _ = model.forward_pass(training_data[i:end], training_label[i:end])
                    model.backward_pass(training_data[i:end], training_label[i:end])
            # finish training

            train_trials.append(model.get_acc(training_data, training_label))
            test_trials.append(model.get_acc(validation_data, validation_label))

        model.get_confusion_matrix(validation_data, validation_label, output_size)
        model.get_metrics(testing_data, testing_label)


        summary = get_trials_summary(train_trials, test_trials)
        avg.append([summary[0], summary[1]])
        std.append([summary[2], summary[3]])

    plot_trials(avg, std, num_epochs_set)


if __name__ == "__main__":
    main()