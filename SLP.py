"""A beginner-friendly single-layer perceptron (one simple neural network)."""  # Explain what this file demonstrates.


def step_function(value):  # Define the activation function used by the neuron.
	if value >= 0:  # Check whether the neuron's weighted total reaches the threshold.
		return 1  # Return class 1 when the threshold is reached.
	return 0  # Return class 0 when the threshold is not reached.


def predict(inputs, weights, bias):  # Define a function that makes one prediction.
	weighted_total = 0  # Start the weighted total at zero.
	for input_value, weight in zip(inputs, weights):  # Pair each input with its weight.
		weighted_total += input_value * weight  # Add this input's weighted contribution.
	weighted_total += bias  # Add the bias so the neuron can shift its threshold.
	return step_function(weighted_total)  # Convert the total into either 0 or 1.


def train(inputs_list, expected_outputs, learning_rate, epochs):  # Define the learning process.
	weights = [0.0, 0.0]  # Create one weight for each of the two input features.
	bias = 0.0  # Start with no bias.
	for epoch in range(epochs):  # Repeat training for the requested number of rounds.
		for inputs, expected_output in zip(inputs_list, expected_outputs):  # Train on each example.
			prediction = predict(inputs, weights, bias)  # Ask the neuron for its current answer.
			error = expected_output - prediction  # Measure how far the answer is from the target.
			for index in range(len(weights)):  # Visit each weight one at a time.
				weights[index] += learning_rate * error * inputs[index]  # Adjust the weight using the error.
			bias += learning_rate * error  # Adjust the bias using the same error.
	return weights, bias  # Give back the learned values.


training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Store the four possible pairs of binary inputs.
expected_outputs = [0, 0, 0, 1]  # Store the correct AND-gate answer for each pair.
learning_rate = 0.1  # Choose how large each learning adjustment should be.
epochs = 10  # Choose how many times the network sees all training examples.

learned_weights, learned_bias = train(training_inputs, expected_outputs, learning_rate, epochs)  # Train the neuron.

print("Learned weights:", learned_weights)  # Display the weights learned during training.
print("Learned bias:", learned_bias)  # Display the bias learned during training.
print("Predictions:")  # Add a heading before showing predictions.
for inputs in training_inputs:  # Make a prediction for every input pair.
	prediction = predict(inputs, learned_weights, learned_bias)  # Use the trained neuron on this pair.
	print(inputs, "->", prediction)  # Display the input pair and the predicted output.
