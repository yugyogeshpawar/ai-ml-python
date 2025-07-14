# Part 3, Topic 4: GPU and CUDA

Modern deep learning models are computationally intensive. Training them on a CPU can be incredibly slow. To accelerate the process, we use **Graphics Processing Units (GPUs)**. GPUs are specialized hardware designed for parallel computations, which is exactly what's needed for the matrix multiplications that are at the heart of deep learning.

**CUDA** is a parallel computing platform and programming model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose computing. PyTorch has deep integration with CUDA, making it easy to move computations from the CPU to the GPU.

## The `.to(device)` Method

The key to using the GPU in PyTorch is the `.to(device)` method. This method is available on all `nn.Module`s and `torch.Tensor`s. It sends the object to the specified device (e.g., a specific GPU or the CPU).

### The Standard Workflow

The standard workflow for using the GPU is as follows:

**1. Define the Device:**
At the beginning of your script, you should define a `device` object. This code checks if a CUDA-enabled GPU is available and falls back to the CPU if it's not. This makes your code portable.

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```
If you have multiple GPUs, you can specify which one to use, e.g., `torch.device('cuda:0')` for the first GPU, `torch.device('cuda:1')` for the second, and so on.

**2. Move Your Model to the Device:**
After you create an instance of your model, you must move all of its parameters and buffers to the GPU.

```python
model = MyModel()
model.to(device)
```
This operation is done **in-place**.

**3. Move Your Data to the Device:**
This is the most critical step in the training loop. For every batch of data you get from your `DataLoader`, you must move the input tensors and target tensors to the same device as your model.

```python
# Inside your training loop
for inputs, labels in dataloader:
    # Move tensors to the configured device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # ... rest of the training loop ...
```

### Why is this necessary?

An operation between two tensors can only happen if they are on the **same device**. If you try to pass a CPU tensor (`inputs`) to a model that is on the GPU (`model`), you will get a `RuntimeError`.

By moving both the model and the data to the `device`, you ensure that all computations in the forward pass happen on the GPU, which provides a massive speedup.

## Common Pitfalls

-   **Forgetting to move data:** The most common error is forgetting to move the `inputs` and `labels` tensors to the `device` inside the training loop. This will immediately raise a `RuntimeError` about mismatched tensor devices.
-   **CPU-specific operations:** Some operations, like converting a tensor to a NumPy array (`.numpy()`), can only be done on CPU tensors. If you have a tensor on the GPU, you must first move it back to the CPU before converting it.
    ```python
    gpu_tensor = torch.randn(3, 3).to(device)

    # This will fail:
    # numpy_array = gpu_tensor.numpy()

    # Correct way:
    cpu_tensor = gpu_tensor.cpu()
    numpy_array = cpu_tensor.numpy()
    ```
-   **Checking for GPU availability:** Always use the `torch.cuda.is_available()` check. This ensures your code will still run on machines that don't have a compatible GPU, making your work more accessible and reproducible.

## Data Parallelism (`nn.DataParallel`)

If you have access to multiple GPUs on a single machine, you can further accelerate training using `nn.DataParallel`. This module automatically splits the data in a batch across your available GPUs, copies the model to each GPU, and then fuses the results.

Using it is straightforward:
```python
model = MyModel()
if torch.cuda.device_count() > 1:
  print(f"Using {torch.cuda.device_count()} GPUs!")
  model = nn.DataParallel(model)

model.to(device)
```
`DataParallel` handles the distribution and synchronization for you. However, for more advanced multi-GPU and multi-machine training, the `torch.distributed` package is now the recommended approach.

The `gpu_and_cuda_example.py` script demonstrates the standard workflow for moving a model and data to the GPU.
