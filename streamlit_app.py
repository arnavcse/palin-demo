import streamlit as st
import numpy as np
import pickle
from neural_network import NeuralNetwork

# Define the function to classify user input
def classify_user_input(user_input, nn):
    if len(user_input) != 10 or not all(bit in '01' for bit in user_input):
        return "Invalid input. Please enter a 10-bit binary string."
    binary = np.array(list(user_input), dtype=int)
    prediction = nn.forward(binary)
    if prediction >= 0.5:
        return "Palindrome"
    else:
        return "Not Palindrome"

def main():
    st.title("Palindrome Classifier with Neural Network")

    # Load the trained model from the pickle file
    pickle_file_path = "/Users/arnav/Desktop/DL NLP Assignment 1/Working/neural_network_model.pkl"
    with open(pickle_file_path, 'rb') as f:
        nn = pickle.load(f)

    # User input
    user_input = st.text_input("Enter a 10-bit binary string (e.g., 1010101010):", "")

    # Classification button
    if st.button("Classify"):
        result = classify_user_input(user_input, nn)
        st.write(f"Classification result: {result}")

if __name__ == "__main__":
    main()
