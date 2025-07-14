# linear_regression.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Prepare the Data ---

# Create synthetic data for a linear relationship
# y = w * x + b
# We'll use w = 2 and b = 5 as our ground truth
true_weight = 2.0
true_bias = 5.0
num_samples = 100

# Create input features (x)
X = torch.randn(num_samples, 1) * 10  # Shape (100, 1)

# Create target values (y) with some noise
noise = torch.randn(num_samples, 1) * 2 # Add some randomness
y = true_weight * X + true_bias + noise # Shape (100, 1)

# --- 2. Define the Model ---

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # A linear layer is all we need for linear regression
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Model parameters
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# --- 3. Define Loss Function and Optimizer ---

# Mean Squared Error is a common loss function for regression
loss_function = nn.MSELoss()

# Stochastic Gradient Descent (SGD) is used to update the parameters
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# --- 4. The Training Loop ---

num_epochs = 100

print("--- Starting Training ---")
for epoch in range(num_epochs):
    # Forward pass: compute predicted y by passing x to the model
    y_predicted = model(X)

    # Compute loss
    loss = loss_function(y_predicted, y)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Update weights: call step() to update the parameters
    optimizer.step()

    # Zero gradients: clear the gradients of all optimized tensors
    optimizer.zero_grad()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- Training Finished ---")

# --- 5. Visualize the Results ---

# Get the learned parameters
learned_weight, learned_bias = model.parameters()
print(f"\nLearned Weight: {learned_weight.item():.3f}, Learned Bias: {learned_bias.item():.3f}")
print(f"True Weight: {true_weight}, True Bias: {true_bias}")

# Detach tensors from the computation graph for plotting
predicted = model(X).detach().numpy()
X_numpy = X.detach().numpy()
y_numpy = y.detach().numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Original data', alpha=0.7)
plt.plot(X_numpy, predicted, color='red', linewidth=3, label='Fitted line')
plt.title('Linear Regression Fit')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.legend()
plt.grid(True)
plt.show()
