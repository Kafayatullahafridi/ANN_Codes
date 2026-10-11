"""Train a neural network on scikit-learn's bundled diabetes dataset."""  # Describe the purpose of this file.

import math  # Import math so we can calculate the root mean squared error.

try:  # Try to import the machine-learning tools needed by this example.
	from sklearn.datasets import load_diabetes  # Import the bundled diabetes regression dataset.
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Import regression evaluation measures.
	from sklearn.model_selection import train_test_split  # Import the tool that separates training and testing data.
	from sklearn.neural_network import MLPRegressor  # Import a multi-layer neural network for number prediction.
	from sklearn.pipeline import Pipeline  # Import a tool that joins preprocessing and model training safely.
	from sklearn.preprocessing import StandardScaler  # Import a tool that puts features on similar scales.
except ImportError as error:  # Catch the error when scikit-learn is not installed.
	raise SystemExit("Install scikit-learn first with: python -m pip install scikit-learn") from error  # Show a beginner-friendly installation command.


def main():  # Define the main function that runs the complete machine-learning workflow.
	dataset = load_diabetes()  # Load the regression dataset bundled with scikit-learn.
	target_unit = "diabetes target units"  # Identify the native units used by the dataset target.
	input_data = dataset.data  # Store the ten medical features used as neural-network inputs.
	target_data = dataset.target  # Store the disease-progression values the network must predict.

	training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(  # Split examples into training and testing groups.
		input_data,  # Provide all input features to the splitting tool.
		target_data,  # Provide all target values to the splitting tool.
		test_size=0.2,  # Reserve 20 percent of the examples for final testing.
		random_state=42,  # Make the same examples go into each group every time.
	)  # Finish the train-test split.

	neural_network = Pipeline(  # Create a safe sequence of preprocessing followed by neural-network training.
		[  # Start the list of steps in the pipeline.
			("scale_features", StandardScaler()),  # Standardize values using training data only.
			(  # Start the neural-network step.
				"model",  # Give the neural-network step a readable name.
				MLPRegressor(  # Create a multi-layer neural network for regression.
					hidden_layer_sizes=(32, 16),  # Use two hidden layers with 32 and 16 neurons.
					activation="relu",  # Use the ReLU activation function in the hidden layers.
					solver="adam",  # Use the Adam algorithm to update the network weights.
					learning_rate_init=0.001,  # Set the initial size of each weight update.
					max_iter=1000,  # Allow enough training rounds for the network to converge.
					early_stopping=True,  # Stop when validation performance stops improving.
					validation_fraction=0.1,  # Use 10 percent of training data for early stopping validation.
					n_iter_no_change=15,  # Stop after 15 rounds without meaningful improvement.
					random_state=42,  # Make the network's starting values repeatable.
				),  # Finish creating the neural network.
			),  # Finish the neural-network pipeline step.
		]  # Finish the pipeline step list.
	)  # Finish creating the complete pipeline.

	print("Training the neural network...")  # Tell the user that training has started.
	neural_network.fit(training_inputs, training_targets)  # Learn patterns from training data only.
	predictions = neural_network.predict(testing_inputs)  # Predict target values for data the model has not seen.

	mean_absolute_error_value = mean_absolute_error(testing_targets, predictions)  # Calculate the average absolute prediction error.
	mean_squared_error_value = mean_squared_error(testing_targets, predictions)  # Calculate the average squared prediction error.
	root_mean_squared_error_value = math.sqrt(mean_squared_error_value)  # Convert squared error back to the target's original scale.
	r2_value = r2_score(testing_targets, predictions)  # Calculate how much variation the model explains.

	print("\nEvaluation results:")  # Print a heading for the model's test results.
	print(f"Mean absolute error: {mean_absolute_error_value:.3f} ({target_unit})")  # Print the average absolute error.
	print(f"Root mean squared error: {root_mean_squared_error_value:.3f} ({target_unit})")  # Print the typical larger-error-sensitive error.
	print(f"R-squared score: {r2_value:.3f}")  # Print the proportion of target variation explained by the model.

	print("\nSample predictions:")  # Print a heading for a few individual predictions.
	for actual_value, predicted_value in zip(testing_targets[:5], predictions[:5]):  # Visit the first five test examples.
		print(f"Actual: {actual_value:.3f} | Predicted: {predicted_value:.3f}")  # Show actual and predicted diabetes targets.


if __name__ == "__main__":  # Check whether this file was run directly rather than imported.
	main()  # Start the complete training and evaluation workflow.
