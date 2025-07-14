# Part 2, Topic 4: Optimizers and the Training Loop

You've learned how to define a model, prepare data, and calculate the loss. The final piece of the puzzle is the **optimizer**. The optimizer is the algorithm that adjusts the model's parameters (weights and biases) based on the gradients computed during backpropagation, with the goal of minimizing the loss.

## The Role of the Optimizer

After `loss.backward()` computes the gradients (`param.grad`) for all parameters, the optimizer's job is to use these gradients to update the parameters' values. This update step is called **optimization**.

The core idea is based on **gradient descent**. The gradient of the loss function with respect to a parameter tells us the direction of the steepest ascent. To minimize the loss, we should move the parameter's value in the **opposite** direction of the gradient.

The basic update rule for a parameter `W` is:
`W_new = W_old - learning_rate * W_gradient`

-   `learning_rate`: A small scalar value that controls how big of a step the optimizer takes. Choosing a good learning rate is critical for effective training.

## The `torch.optim` Module

PyTorch provides a variety of optimization algorithms in the `torch.optim` module. You don't need to implement the update rules yourself.

### Creating an Optimizer

To create an optimizer, you need to give it the model's parameters that it should optimize and specify the learning rate.

```python
import torch.optim as optim

# model.parameters() tells the optimizer which tensors to update
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### Common Optimizers

1.  **`optim.SGD` (Stochastic Gradient Descent):**
    -   The classic, most basic optimization algorithm.
    -   It can be effective but may be slow to converge.
    -   Often used with **momentum**, a technique that helps the optimizer accelerate in the correct direction and dampens oscillations.
    -   `optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`

2.  **`optim.Adam` (Adaptive Moment Estimation):**
    -   One of the most popular and effective optimizers. It's an excellent default choice for many problems.
    -   It adapts the learning rate for each parameter individually, using estimates of the first and second moments of the gradients. This often leads to faster convergence than standard SGD.
    -   `optimizer = optim.Adam(model.parameters(), lr=0.001)`

3.  **`optim.RMSprop` (Root Mean Square Propagation):**
    -   Another adaptive learning rate optimizer that works well for many problems, especially with RNNs.

## The Complete Training Loop

Now we can put everything together to form the standard PyTorch training loop. This loop is the heart of any model training process.

```python
# Assume model, train_loader, loss_fn, and optimizer are already defined

num_epochs = 10
model.train() # Set the model to training mode

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # 1. Move data to the target device (e.g., GPU)
        data = data.to(device)
        targets = targets.to(device)

        # 2. Forward Pass: Get model predictions
        scores = model(data)
        loss = loss_fn(scores, targets)

        # 3. Backward Pass: Compute gradients
        # First, zero out gradients from the previous step
        optimizer.zero_grad()
        loss.backward()

        # 4. Update Parameters: Tell the optimizer to take a step
        optimizer.step()

        # Optional: Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
```

### Breakdown of the Loop Steps

1.  **Get Data:** The `DataLoader` provides a batch of data and corresponding targets.
2.  **Forward Pass:** The data is passed through the model to get predictions (`scores`). The loss is calculated by comparing these scores to the true `targets`.
3.  **`optimizer.zero_grad()`:** This is a critical step. As we learned previously, `.backward()` accumulates gradients. We must clear the old gradients before computing the new ones for the current batch.
4.  **`loss.backward()`:** This computes the gradient of the loss with respect to every model parameter that has `requires_grad=True`.
5.  **`optimizer.step()`:** This is where the optimizer performs the update. It iterates through all the parameters it was given at initialization and updates their values using the computed gradients and its internal update rule (e.g., the Adam or SGD update rule).

This loop represents one full pass over the training dataset, known as an **epoch**. This process is repeated for multiple epochs until the model's performance on a validation set stops improving.

The `optimizers_example.py` script provides a runnable example of this full training loop.
