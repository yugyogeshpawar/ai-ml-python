# Exercises: Autograd and Automatic Differentiation

These exercises will help you solidify your understanding of how `autograd` works.

## Exercise 1: Manual Gradient Calculation vs. Autograd

**Task:** For the equation `y = 3x^2 + 2`, calculate the gradient of `y` with respect to `x` at the point `x = 4`.

1.  **Manually:** Calculate the derivative dy/dx by hand. Then, substitute `x = 4` to find the value of the gradient.
2.  **With Autograd:** Write a PyTorch script to compute the same gradient.
    -   Create a tensor `x` with the value `4.0` and `requires_grad=True`.
    -   Define the equation `y = 3 * x**2 + 2`.
    -   Call `y.backward()`.
    -   Print the gradient stored in `x.grad`.

**Goal:** Verify that `autograd` computes the correct gradient, confirming your understanding of both the calculus and the PyTorch implementation.

## Exercise 2: Gradients of a Multi-Output Model

**Task:** Consider a model with one input `x` and two outputs, `y` and `z`.
-   `y = x * 2`
-   `z = x * 3`

The final loss is the sum of `y` and `z`.
-   `loss = y + z`

Calculate the gradient of the `loss` with respect to `x` when `x = 2`.

1.  Create a tensor `x` with the value `2.0` and `requires_grad=True`.
2.  Define `y`, `z`, and `loss`.
3.  Call `loss.backward()`.
4.  Print `x.grad`.

**Goal:** Understand how `autograd` handles computations involving multiple paths from an input to the final loss. The chain rule dictates that the gradients from each path should be summed. What do you expect the gradient to be?

## Exercise 3: The Importance of `grad.zero_()`

**Task:** Write a script that demonstrates why you must zero out gradients in a training loop.

1.  Create a parameter `w` with `requires_grad=True`.
2.  Simulate a simple training loop that runs for 2 iterations.
3.  In each iteration:
    -   Calculate a `loss` (e.g., `loss = w * 2`).
    -   Call `loss.backward()`.
    -   Print the value of `w.grad`.
    -   **Do not** zero out the gradient.

Run the script and observe the value of `w.grad` in the second iteration. Now, add `w.grad.zero_()` at the end of the loop and run it again.

**Goal:** See firsthand how gradients accumulate if they are not explicitly reset. This will reinforce why `optimizer.zero_grad()` is a critical step in every PyTorch training loop.
