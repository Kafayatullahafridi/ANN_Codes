import numpy as np
import pandas as pd

# -------------------------------------------------
# 1. Load real data (Iris dataset)
# -------------------------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, header=None, names=columns)

print("Dataset shape:", df.shape)
print(df.head())
print(df["species"].value_counts())

# -------------------------------------------------
# 2. Preprocess with pandas + numpy
# -------------------------------------------------
# Encode labels (setosa=0, versicolor=1, virginica=2)
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df["label"] = df["species"].map(label_map)

# Features and one-hot labels
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float64)
y = df["label"].values

# One-hot encode y
y_onehot = np.eye(3)[y]

# Train / test split (80/20) – pure numpy
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

# Standardize features (important for NN)
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -------------------------------------------------
# 3. Neural Network from scratch (NumPy only)
# -------------------------------------------------
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        # Output layer gradient
        dz2 = (y_pred - y_true) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=2000, print_every=200):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))  # cross-entropy
            self.backward(X, y, y_pred)

            if epoch % print_every == 0 or epoch == epochs - 1:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# -------------------------------------------------
# 4. Train the network
# -------------------------------------------------
nn = SimpleNN(input_size=4, hidden_size=16, output_size=3, lr=0.05)
nn.train(X_train, y_train, epochs=2500, print_every=250)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
y_pred_test = nn.predict(X_test)
y_true_test = np.argmax(y_test, axis=1)

test_acc = np.mean(y_pred_test == y_true_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Simple confusion matrix with pandas
cm = pd.crosstab(
    pd.Series(y_true_test, name="Actual"),
    pd.Series(y_pred_test, name="Predicted")
)
print("\nConfusion Matrix:")
print(cm)