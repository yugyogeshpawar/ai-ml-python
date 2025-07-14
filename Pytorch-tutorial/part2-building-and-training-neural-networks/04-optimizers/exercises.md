# Exercises: Optimizers and the Training Loop

These exercises focus on the components of the training loop and how to use different optimizers.

## Exercise 1: The Three Key Optimizer Steps

**Task:** In a standard PyTorch training loop, there are three lines of code related to the optimizer and the loss that must be called in a specific order.

1.  List the three method calls.
2.  Briefly explain what each one does.
3.  Explain why their order (`zero_grad`, `backward`, `step`) is critical. What would happen if you called `step` before `backward`?

**Goal:** Solidify your memory of the fundamental training loop structure, which is the backbone of all model training in PyTorch.

## Exercise 2: Try a Different Optimizer

**Task:** Modify the `optimizers_example.py` script to use the `optim.SGD` optimizer instead of `optim.Adam`.

1.  Import `torch.optim as optim`.
2.  Find the line where the optimizer is created: `optimizer = optim.Adam(...)`.
3.  Change it to `optimizer = optim.SGD(model.parameters(), lr=0.01)`. Note that SGD often requires a larger learning rate than Adam, so we've changed it from `0.001` to `0.01`.
4.  Run the script for a few epochs.

**Goal:** Practice swapping out different optimizers. This is a common hyperparameter to tune when trying to improve model performance. Do you notice a difference in how quickly the loss decreases compared to Adam?

## Exercise 3: What Does `model.parameters()` Do?

**Task:** Write a short script to investigate the output of `model.parameters()`.

1.  Instantiate the `SimpleCNN` model from the example.
2.  Create an `optim.Adam` optimizer, passing `model.parameters()` to it.
3.  Loop through `model.parameters()` and print the shape of each parameter tensor.
4.  Compare this to the output of looping through `model.named_parameters()`.

**Goal:** Understand that `model.parameters()` provides the optimizer with an iterable of all learnable tensors (weights and biases) in the model, which is how the optimizer knows what to update.
