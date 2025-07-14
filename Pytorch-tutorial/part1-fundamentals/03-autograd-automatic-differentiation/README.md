# Part 1, Topic 3: Autograd and Automatic Differentiation

Now that you understand PyTorch Tensors, let's explore one of the most critical features of PyTorch for neural networks: `torch.autograd`, the automatic differentiation engine.

## What is Automatic Differentiation?

Training a neural network involves an algorithm called **backpropagation**. At its core, backpropagation requires calculating the **gradient** (or derivative) of a loss function with respect to each of the model's parameters (weights and biases). This gradient tells us how to adjust each parameter to reduce the loss.

**Automatic differentiation** is a technique that automatically calculates these gradients for any computational graph. PyTorch's `autograd` engine handles this process for us, so we don't have to compute these complex derivatives by hand.

## The Computation Graph

PyTorch builds a **dynamic computation graph** as you perform operations on tensors. This graph consists of:
-   **Nodes:** The tensors themselves.
-   **Edges:** The functions or operations that produce output tensors from input tensors.

`autograd` uses this graph to track the relationships between tensors. When we want to compute gradients, it traverses this graph backward from the final output (usually the loss) to each of the input tensors that require gradients.

### How it Works

1.  **Forward Pass:** When you perform operations on tensors, PyTorch builds the computation graph. For a tensor to be part of this graph, its `requires_grad` attribute must be set to `True`.
2.  **Backward Pass:** When you call `.backward()` on a scalar tensor (e.g., the loss), `autograd` traverses the graph backward. It uses the **chain rule** of calculus to compute the gradient of that scalar with respect to every tensor in the graph that has `requires_grad=True`.
3.  **Storing Gradients:** The computed gradients are accumulated in the `.grad` attribute of each respective tensor.

## A Simple Example

Let's see `autograd` in action. We'll create a simple computation: `y = w * x + b`, where `w` and `b` are parameters we want to optimize.

```python
import torch

# Create tensors with requires_grad=True to track computation
x = torch.ones(5)  # Input tensor
y_true = torch.zeros(5) # Expected output
w = torch.randn(5, requires_grad=True)
b = torch.randn(5, requires_grad=True)

# Build a simple model
y_pred = w * x + b
loss = torch.mean((y_pred - y_true) ** 2) # Mean Squared Error loss

print(f"Predicted y: {y_pred}")
print(f"Loss: {loss}")
```

At this point, a computation graph has been created that connects `w` and `b` to the `loss`.

### Computing Gradients

Now, let's compute the gradients of the `loss` with respect to `w` and `b`.

```python
# Perform backpropagation
loss.backward()

# The gradients are now stored in the .grad attribute of w and b
print(f"Gradient of loss w.r.t. w: {w.grad}")
print(f"Gradient of loss w.r.t. b: {b.grad}")
```

The `w.grad` and `b.grad` tensors now hold the values of ∂(loss)/∂w and ∂(loss)/∂b, respectively. An optimizer would use these gradients to update the values of `w` and `b`.

## Disabling Gradient Tracking

By default, tensors with `requires_grad=True` will track their history. There are times when you don't need this, for example, during **inference** (when you are only evaluating a trained model) or when you want to update model parameters without `autograd` tracking the update itself.

You can disable gradient tracking in a few ways:

### 1. Using `torch.no_grad()`

This is the recommended way to disable gradient tracking for a block of code. It's a context manager that tells PyTorch not to build the computation graph within that block.

```python
print(w.requires_grad) # True
with torch.no_grad():
    # Operations inside this block will not be tracked
    new_y = w * x + b
    print(new_y.requires_grad) # False
```

### 2. Using `.detach()`

This method creates a new tensor that shares the same data as the original tensor but is detached from the computation graph. It doesn't require gradients.

```python
detached_w = w.detach()
print(detached_w.requires_grad) # False
```

This is useful when you want to use a tensor's value in a computation that should not be part of the main graph.

## Key Concepts to Remember

-   `requires_grad=True`: Tells PyTorch to track operations on this tensor for `autograd`.
-   `.backward()`: Computes the gradients of a scalar tensor with respect to all other tensors in the graph that have `requires_grad=True`.
-   `.grad`: The attribute where the computed gradients are stored.
-   `torch.no_grad()`: A context manager to prevent PyTorch from tracking history, saving memory and computation during inference.

Understanding `autograd` is fundamental to training any neural network in PyTorch. The `autograd_example.py` script provides a runnable demonstration of these concepts.
