# Palindrome Classifier

## Overview
This repository contains the implementation of a Palindrome Classifier using a feedforward neural network. The classifier is trained to identify whether a given 10-bit binary string is a palindrome or not.

## Problem Statement
1. Implement back propagation using only methods that can calculate derivatives.
2. Generate custom data and labels for training.
3. Implement back propagation from scratch (ab initio).
4. Use only one hidden layer with the minimum number of neurons.
5. Design an appropriate architecture for palindrome classification.
6. Train a feedforward network using 1024 input strings labeled as 1 (palindrome) or 0 (non-palindrome).
7. Perform training and testing with 4-fold cross-validation.
8. Measure precision of the classifier.
9. Investigate the behavior of hidden layer neurons.

## Implementation
- **Back Propagation**: Implemented from scratch using NumPy to calculate derivatives.
- **Data Generation**: Custom data and labels are generated for training.
- **Neural Network Architecture**: Utilizes a single hidden layer with minimal neurons.
- **Training and Testing**: 4-fold cross-validation is performed for training and testing.
- **Precision Measurement**: Precision of the classifier is calculated.
- **Hidden Layer Analysis**: Investigates the behavior of hidden layer neurons.

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your-username/palindrome-classifier.git
