# Part 1 Project: Linear Regression From Scratch

This project brings together all the concepts from Part 1—Tensors, Autograd, and `nn.Module`—to build a simple linear regression model from scratch.

## Goal

The goal of this project is to predict a target value based on a single input feature. We will create a synthetic dataset, build a model, train it using gradient descent, and visualize the results.

## What is Linear Regression?

Linear regression is a fundamental machine learning algorithm that models the relationship between a dependent variable (the target) and one or more independent variables (the features) by fitting a linear equation to the observed data.

The equation for a simple linear regression model is:
`y = w * x + b`

-   `y`: The predicted value.
-   `x`: The input feature.
-   `w`: The weight (or slope).
-   `b`: The bias (or y-intercept).

The goal of training is to find the optimal values for `w` and `b` that minimize the difference between the predicted `y` and the true `y`.

## Project Steps

The `linear_regression.py` script follows these steps:

1.  **Prepare the Data:** We will create a synthetic dataset using `torch.randn`. We'll start with a known weight and bias (`w=2`, `b=5`) and add some noise to make the task more realistic.
2.  **Define the Model:** We will create a `LinearRegressionModel` class that inherits from `nn.Module`. This model will contain a single `nn.Linear` layer.
3.  **Define the Loss Function and Optimizer:**
    -   **Loss Function:** We will use Mean Squared Error (`nn.MSELoss`), which is the standard loss function for regression problems. It measures the average squared difference between the predicted and actual values.
    -   **Optimizer:** We will use Stochastic Gradient Descent (`torch.optim.SGD`) to update the model's parameters (`w` and `b`).
4.  **The Training Loop:** This is the core of the project. For a specified number of epochs, we will:
    a.  Perform a **forward pass** to get the model's predictions.
    b.  Calculate the **loss** by comparing the predictions to the true values.
    c.  **Zero the gradients** (`optimizer.zero_grad()`) to prevent accumulation.
    d.  Perform a **backward pass** (`loss.backward()`) to compute the gradients.
    e.  **Update the weights** (`optimizer.step()`) using the computed gradients.
5.  **Visualize the Results:** After training, we will plot the original data points, the true line, and the line our trained model has learned. This will give us a clear visual confirmation of whether our model has learned the underlying relationship.

This project is a complete, end-to-end example of a PyTorch workflow and serves as a strong foundation for the more complex models we will build in Part 2.
