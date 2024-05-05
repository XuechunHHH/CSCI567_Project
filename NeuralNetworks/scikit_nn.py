from sklearn.neural_network import MLPClassifier
from preprocess import preprocess_data

def main():
    training_data, training_label, validation_data, validation_label, testing_data, testing_label = preprocess_data()

    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5)).fit(training_data, training_label)
    print(nn.predict(testing_data))
    print(nn.score(testing_data, testing_label))

if __name__ == "__main__":
    main()