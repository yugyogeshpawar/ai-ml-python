# Part 2, Topic 3: Loss Functions

A **loss function** (or criterion) is one of the most important components of a neural network. It measures how different the model's prediction is from the true target. The goal of training is to adjust the model's parameters to minimize the value of the loss function.

## The Role of the Loss Function

Think of the loss function as a way of scoring how well your model is doing on a particular sample.
-   A **high loss** means the model's prediction was far from the true target.
-   A **low loss** means the model's prediction was close to the true target.

The entire training process is driven by this score. The gradients calculated during the backward pass (`loss.backward()`) are the gradients *of the loss function* with respect to the model's parameters. Therefore, the choice of loss function is critical and depends entirely on the type of problem you are trying to solve.

PyTorch provides a variety of standard loss functions in the `torch.nn` module.

## Common Loss Functions

Let's look at the two most common categories of problems and their associated loss functions.

### 1. For Regression Problems

In regression, the goal is to predict a continuous value (e.g., the price of a house, the temperature tomorrow).

#### `nn.MSELoss` (Mean Squared Error)

This is the most common loss function for regression. It calculates the average of the squared differences between the predicted values and the true values.

**Formula:** `Loss = (1/N) * Σ(y_pred - y_true)²`

-   **When to use it:** It's a great default choice for regression tasks.
-   **Properties:** Squaring the error penalizes larger errors more heavily than smaller ones. It also results in a convex loss landscape, which is easier for optimizers to navigate.

```python
import torch
import torch.nn as nn

loss_fn = nn.MSELoss()

# Example
predicted = torch.tensor([2.5, 4.8, 1.2])
target = torch.tensor([3.0, 5.0, 1.0])

loss = loss_fn(predicted, target)
print(f"MSE Loss: {loss.item()}") # Output: ~0.0967
```

### 2. For Classification Problems

In classification, the goal is to predict a discrete class label (e.g., "cat", "dog", or "bird").

#### `nn.CrossEntropyLoss`

This is the standard loss function for multi-class classification problems. It is highly effective and should be your default choice.

**Important:** `nn.CrossEntropyLoss` in PyTorch cleverly combines two steps in one:
1.  It applies a `LogSoftmax` function to the model's raw output scores (logits).
2.  It then calculates the Negative Log-Likelihood Loss (`NLLLoss`).

Because it includes `LogSoftmax`, you should **not** apply a Softmax or Sigmoid activation function to the final layer of your model before passing its output to this loss function. The model should output raw, unnormalized scores (logits).

-   **When to use it:** For multi-class classification (when an input can belong to one of many classes).
-   **Input Shape:**
    -   `Input` (model's prediction): A tensor of shape `(N, C)`, where `N` is the batch size and `C` is the number of classes. It should contain raw logits.
    -   `Target`: A tensor of shape `(N)` containing the correct class indices (e.g., `0`, `1`, `2`, ...).

```python
loss_fn = nn.CrossEntropyLoss()

# Example: 3 samples, 4 classes
predicted_logits = torch.randn(3, 4) # Raw scores from the model
target_labels = torch.tensor([1, 0, 3]) # Correct class indices

loss = loss_fn(predicted_logits, target_labels)
print(f"Cross-Entropy Loss: {loss.item()}")
```

#### `nn.BCELoss` (Binary Cross-Entropy Loss)

This loss function is used for **binary classification** (when there are only two classes, e.g., 0 or 1).

**Important:** `nn.BCELoss` expects the model's output to be a probability, meaning it should be passed through a **Sigmoid** activation function first to squash the output to the range `[0, 1]`.

-   **When to use it:** For binary classification or multi-label classification (where one sample can belong to multiple classes).
-   **Input Shape:**
    -   `Input` (model's prediction): A tensor of probabilities, shape `(N, *)`.
    -   `Target`: A tensor of the same shape as the input, containing target probabilities (usually 0.0 or 1.0).

```python
loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

# Example: 3 samples, binary classification
predicted_logits = torch.randn(3) # Raw scores
predicted_probs = sigmoid(predicted_logits) # Convert to probabilities
target = torch.tensor([1.0, 0.0, 1.0]) # Target probabilities

loss = loss_fn(predicted_probs, target)
print(f"BCE Loss: {loss.item()}")
```
*Tip: `nn.BCEWithLogitsLoss` combines the `Sigmoid` and `BCELoss` in one step for better numerical stability, similar to how `CrossEntropyLoss` works.*

## Choosing the Right Loss Function

-   **Regression Task?** -> Start with `nn.MSELoss`.
-   **Binary Classification?** -> Use `nn.BCEWithLogitsLoss`.
-   **Multi-Class Classification?** -> Use `nn.CrossEntropyLoss`.

The `loss_functions_example.py` script provides a runnable demonstration of how to use these common loss functions.
