import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class RNN:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros((1, self.hidden_size))

        # Initialize the weight matrices with random values
        self.weights_input = np.random.randn(input_size, hidden_size)
        self.weights_hidden = np.random.randn(hidden_size, hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)

    
    def forward(self, x):
        outputs = np.zeros((x.shape[0], self.output_size))
        return outputs


    def backward(self, inputs, outputs, targets, learning_rate):
        # Initialize the error and gradient arrays with zeros
        error = np.zeros((1, self.output_size))
        hidden_gradient = np.zeros((1, self.hidden_size))
        output_gradient = np.zeros((self.hidden_size, self.output_size))

        # Loop through each output in the sequence in reverse order
        for i in reversed(range(outputs.shape[0])):
            # Compute the error for this output
            error = targets[i] - outputs[i]

            # Compute the output gradient for this output
            print(i)
            output_gradient = np.dot(np.atleast_2d(self.hidden_state.shape).T, np.atleast_2d(error))

            # Add the output gradient to the total output gradient
            self.weights_output =  self.weights_output + (learning_rate * output_gradient)

            # Compute the hidden gradient for this output
            hidden_gradient = np.dot(error, self.weights_output.T) * sigmoid_derivative(self.hidden_state[0])

            # Add the hidden gradient to the total hidden gradient
            self.weights_hidden = self.weights_hidden + learning_rate * np.dot(np.atleast_2d(self.hidden_state[0]).T, np.atleast_2d(hidden_gradient))
            self.weights_input = self.weights_input + learning_rate * np.dot(np.atleast_2d(inputs[i]).T, np.atleast_2d(hidden_gradient))


    def train(self, x, yi, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = 0
            #loss = np.sum((y - y_pred)**2)
            self.backward(x, y_pred, yi, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, loss: {loss}")
