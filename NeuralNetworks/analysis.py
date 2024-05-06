import numpy as np
import matplotlib.pyplot as plt


def get_trials_summary(train_trials, test_trials):
    train_avg = sum(train_trials) / len(train_trials)
    test_avg = sum(test_trials) / len(test_trials)
    train_std = np.std(train_trials)
    test_std = np.std(test_trials)

    return train_avg, test_avg, train_std, test_std

def plot_trials(avg, std, num_epochs_set):
    avg = np.array(avg)
    if std is not None:
        std = np.array(std)

    plt.plot(range(len(avg)), avg[:, 0])
    plt.plot(range(len(avg)), avg[:, 1])
    plt.xticks(range(len(num_epochs_set)), num_epochs_set)
    plt.legend(["training set", "validation set"])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.show()

    if std is not None:
        plt.clf()

        plt.plot(range(len(std)), std[:, 0])
        plt.plot(range(len(std)), std[:, 1])
        plt.xticks(range(len(num_epochs_set)), num_epochs_set)
        plt.legend(["training set", "validation set"])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.show()