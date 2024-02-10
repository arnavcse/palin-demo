import streamlit as st
import numpy as np

# Define the neural network architecture
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

# Function to classify user input
def classify_user_input(user_input, nn):
    if len(user_input) != 10 or not all(bit in '01' for bit in user_input):
        return "Invalid input. Please enter a 10-bit binary string."
    binary = np.array(list(user_input), dtype=int)
    prediction = nn.forward(binary)
    if prediction >= 0.5:
        return "Palindrome"
    else:
        return "Not Palindrome"

# Function to generate data
def generate_data():
    data = []
    for i in range(1024):
        binary = np.array(list(np.binary_repr(i, width=10)), dtype=int)
        is_palindrome = (binary == binary[::-1]).all()
        data.append((binary, is_palindrome))
    return data

# Main function to setup Streamlit UI
def main():
    st.title("Palindrome Classifier with Neural Network")

    # Load and train the neural network
    input_size = 10
    hidden_size = 4
    output_size = 1
    learning_rate = 0.01
    data = generate_data()
    X = np.array([d[0] for d in data], dtype=int)
    y = np.array([d[1] for d in data], dtype=int)
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    epochs = 50000
    nn.train(X, y.reshape(-1, 1), epochs)

    user_input = st.text_input("Enter a 10-bit binary string (e.g., 1010101010):", "")
    if st.button("Classify"):
        result = classify_user_input(user_input, nn)
        st.write(f"Classification result: {result}")

if __name__ == "__main__":
    main()
