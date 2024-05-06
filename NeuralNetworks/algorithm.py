import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

class Algorithm:
    def forward_pass(self):
        pass

    def get_predictions(self, data, label):
        predictions, _ = self.forward_pass(data, label)
        return predictions

    def get_acc(self, data, label):
        predictions, _ = self.forward_pass(data, label)
        rows = len(data)

        result = np.count_nonzero(predictions == label)

        # return accuracy
        return result / rows
    
    def get_confusion_matrix(self, data, label, size):
        matrix = np.zeros((size, size), dtype=int)

        predictions, _ = self.forward_pass(data, label)
        rows = len(data)

        for i in range(rows):
            matrix[predictions[i]][label[i]] += 1

        sn.heatmap(matrix, annot=True)
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.show()

    def get_metrics(self, data, label):
        predictions = self.get_predictions(data, label)

        accuracy = self.get_acc(data, label)
        precision = precision_score(label, predictions, average='macro')
        recall = recall_score(label, predictions, average='macro')
        f1 = f1_score(label, predictions, average='macro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)