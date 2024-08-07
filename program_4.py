#program - 4

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def feedforward(X, weights1, bias1, weights2, bias2):
    hidden_input = np.dot(X, weights1) + bias1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights2) + bias2
    final_output = sigmoid(final_input)
    return hidden_output, final_output

def backpropagation(X, y, weights1, bias1, weights2, bias2, hidden_output, final_output, learning_rate):
    # Calculate the error
    error = y - final_output
    d_final_output = error * sigmoid_derivative(final_output)

    # Backpropagate the error
    error_hidden_layer = d_final_output.dot(weights2.T)
    d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_output)

    weights2 += hidden_output.T.dot(d_final_output) * learning_rate
    bias2 += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate
    weights1 += X.T.dot(d_hidden_output) * learning_rate
    bias1 += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

    return weights1, bias1, weights2, bias2

def train(X, y, weights1, bias1, weights2, bias2, learning_rate, epochs):
    for epoch in range(epochs):
        hidden_output, final_output = feedforward(X, weights1, bias1, weights2, bias2)
        weights1, bias1, weights2, bias2 = backpropagation(X, y, weights1, bias1, weights2, bias2, hidden_output, final_output, learning_rate)
        if epoch % 1000 == 0:
            loss = np.mean((y - final_output) ** 2)
            print(f'Epoch {epoch}, Loss: {loss}')
    return weights1, bias1, weights2, bias2

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    input_layer_neurons = 2
    hidden_layer_neurons = 2
    output_layer_neurons = 1

    weights1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    bias1 = np.random.uniform(size=(1, hidden_layer_neurons))
    weights2 = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
    bias2 = np.random.uniform(size=(1, output_layer_neurons))

    learning_rate = 0.1
    epochs = 10000

    weights1, bias1, weights2, bias2 = train(X, y, weights1, bias1, weights2, bias2, learning_rate, epochs)

    _, final_output = feedforward(X, weights1, bias1, weights2, bias2)
    print("Predicted Output:")
    print(final_output)
    print("Actual Output:")
    print(y)
