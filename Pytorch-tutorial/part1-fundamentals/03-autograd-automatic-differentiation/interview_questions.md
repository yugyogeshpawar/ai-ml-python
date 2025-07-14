# Interview Questions: Autograd

These questions probe your understanding of the mechanics and implications of automatic differentiation in PyTorch.

### 1. What is the purpose of the `loss.backward()` call in a PyTorch training loop? What happens internally when it is called?

**Answer:**
The `loss.backward()` call is the core of the learning step in PyTorch. Its purpose is to compute the gradients of the `loss` with respect to all model parameters (tensors with `requires_grad=True`) that were involved in its computation.

Internally, when `loss.backward()` is called:
1.  PyTorch's `autograd` engine starts from the `loss` tensor and traverses the computation graph **backward**.
2.  It applies the **chain rule** of calculus at each step to compute the gradient of the loss with respect to each tensor in the graph.
3.  The calculated gradients for each parameter are **accumulated** (summed) into their respective `.grad` attributes. For example, `w.grad` will hold the value of ∂(loss)/∂w.

These gradients are then used by an optimizer (like `torch.optim.Adam`) to update the model's parameters in a way that minimizes the loss.

### 2. Explain the difference between `torch.no_grad()` and `tensor.detach()`. When would you use each?

**Answer:**
Both `torch.no_grad()` and `.detach()` are used to remove a tensor from `autograd`'s tracking, but they operate differently.

-   `torch.no_grad()` is a **context manager**. Any code executed inside the `with torch.no_grad():` block will not have its operations tracked. This is the standard and most efficient way to disable gradient calculation for a section of code.
    -   **Use Case:** Its primary use is for the **inference/evaluation loop** of a model. During inference, you are not training the model, so there is no need to compute gradients. Wrapping the evaluation code in `torch.no_grad()` saves significant memory and computation time.

-   `tensor.detach()` is a **tensor method**. It creates a new tensor that shares the same data as the original tensor but is "detached" from the current computation graph. It will have `requires_grad=False`.
    -   **Use Case:** This is useful when you need to use a tensor's value in a computation that you do not want to be part of the main backpropagation graph. For example, if you want to plot a tensor's value during training or use it for some calculation that should not affect the gradients of your main model.

**In summary:** Use `torch.no_grad()` for entire blocks of code where you don't need gradients (like evaluation). Use `.detach()` when you need a specific tensor to be excluded from gradient tracking while other computations continue as normal.

### 3. Why do we need to call `optimizer.zero_grad()` at the start of each training iteration? What would happen if we didn't?

**Answer:**
We must call `optimizer.zero_grad()` at the start of each training iteration because the `.backward()` function **accumulates** gradients. It does not overwrite them.

When `loss.backward()` is called, the gradients for each parameter are calculated and **added** to the values already stored in the parameter's `.grad` attribute.

**What would happen if we didn't call `optimizer.zero_grad()`?**
If we did not zero out the gradients, the gradients from the current batch would be added to the gradients from all previous batches. This would mean the optimizer would be performing a parameter update based on a mixture of old and new gradient information, which is incorrect. The learning process would be unstable and would not converge correctly, as the update step would not accurately reflect the error from the current batch.

Therefore, `optimizer.zero_grad()` is a critical step to ensure that the parameter updates are based only on the gradients of the current batch of data.
