import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_output) + self.bias_output)
        return self.output

    def backward(self, X, y, output):
        self.error = y - output
        delta_output = self.error * self.sigmoid_derivative(output)
        self.error_hidden = delta_output.dot(self.weights_output.T)
        delta_hidden = self.error_hidden * self.sigmoid_derivative(self.hidden_output)

        self.weights_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X_train, y_train, epochs):
        for _ in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, output)
