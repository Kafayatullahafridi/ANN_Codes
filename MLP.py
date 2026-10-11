"""A beginner-friendly multi-layer perceptron that learns the XOR problem."""  # Explain what this file demonstrates.


import math  # Import math so we can use the exponential function.
import random  # Import random so the network starts with different weights.


def sigmoid(value):  # Define an activation function that changes any number into a value between 0 and 1.
	return 1 / (1 + math.exp(-value))  # Calculate and return the sigmoid result.


def sigmoid_derivative(value):  # Define the derivative needed to adjust weights during learning.
	return value * (1 - value)  # Calculate the slope of the sigmoid using its output value.


def create_network():  # Define a function that creates the network's weights and biases.
	hidden_weights = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]  # Create weights from two inputs to two hidden neurons.
	hidden_biases = [random.uniform(-1, 1) for _ in range(2)]  # Create one bias for each hidden neuron.
	output_weights = [random.uniform(-1, 1) for _ in range(2)]  # Create weights from the two hidden neurons to the output neuron.
	output_bias = random.uniform(-1, 1)  # Create the output neuron's bias.
	return hidden_weights, hidden_biases, output_weights, output_bias  # Return all network parameters.


def predict(inputs, hidden_weights, hidden_biases, output_weights, output_bias):  # Define a function that makes one prediction.
	hidden_outputs = []  # Create an empty list for the hidden-layer results.
	for neuron_weights, neuron_bias in zip(hidden_weights, hidden_biases):  # Process each hidden neuron.
		weighted_total = neuron_bias  # Start this hidden neuron's total with its bias.
		for input_value, weight in zip(inputs, neuron_weights):  # Pair each input with its hidden-layer weight.
			weighted_total += input_value * weight  # Add the input's weighted contribution.
		hidden_outputs.append(sigmoid(weighted_total))  # Activate the hidden neuron and save its output.
	output_total = output_bias  # Start the output neuron's total with its bias.
	for hidden_output, weight in zip(hidden_outputs, output_weights):  # Pair each hidden output with its output-layer weight.
		output_total += hidden_output * weight  # Add the hidden neuron's weighted contribution.
	output = sigmoid(output_total)  # Activate the output neuron.
	return hidden_outputs, output  # Return both hidden values and the final prediction.


def train(inputs_list, expected_outputs, learning_rate, epochs):  # Define the backpropagation learning process.
	hidden_weights, hidden_biases, output_weights, output_bias = create_network()  # Create randomly initialized parameters.
	for epoch in range(epochs):  # Repeat training for the requested number of rounds.
		for inputs, expected_output in zip(inputs_list, expected_outputs):  # Train on every example in the dataset.
			hidden_outputs, output = predict(inputs, hidden_weights, hidden_biases, output_weights, output_bias)  # Make a prediction.
			output_error = expected_output - output  # Measure the output neuron's prediction error.
			output_delta = output_error * sigmoid_derivative(output)  # Calculate how strongly the output neuron should change.
			old_output_weights = output_weights[:]  # Copy old weights so hidden errors use the previous values.
			for index in range(len(output_weights)):  # Visit each output-layer weight.
				output_weights[index] += learning_rate * output_delta * hidden_outputs[index]  # Adjust the output weight.
			output_bias += learning_rate * output_delta  # Adjust the output neuron's bias.
			for hidden_index in range(len(hidden_weights)):  # Visit each hidden neuron.
				hidden_error = output_delta * old_output_weights[hidden_index]  # Send part of the output error backward.
				hidden_delta = hidden_error * sigmoid_derivative(hidden_outputs[hidden_index])  # Calculate this hidden neuron's adjustment.
				for input_index in range(len(hidden_weights[hidden_index])):  # Visit each input-to-hidden weight.
					hidden_weights[hidden_index][input_index] += learning_rate * hidden_delta * inputs[input_index]  # Adjust the hidden weight.
				hidden_biases[hidden_index] += learning_rate * hidden_delta  # Adjust the hidden neuron's bias.
	return hidden_weights, hidden_biases, output_weights, output_bias  # Return the trained network parameters.


training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Store the four possible pairs of binary inputs.
expected_outputs = [0, 1, 1, 0]  # Store the correct XOR answer for each input pair.
learning_rate = 1.0  # Choose how large each learning adjustment should be.
epochs = 10000  # Choose how many times the network sees all training examples.
random.seed(1)  # Use a fixed starting point so the example gives repeatable results.

trained_network = train(training_inputs, expected_outputs, learning_rate, epochs)  # Train the multi-layer network.

print("Predictions for XOR:")  # Add a heading before showing predictions.
for inputs in training_inputs:  # Make a prediction for every input pair.
	_, output = predict(inputs, *trained_network)  # Use the trained network and ignore the hidden values here.
	prediction = 1 if output >= 0.5 else 0  # Convert the decimal output into class 0 or class 1.
	print(inputs, "->", prediction, "(raw output:", round(output, 3), ")")  # Display the input and its prediction.
