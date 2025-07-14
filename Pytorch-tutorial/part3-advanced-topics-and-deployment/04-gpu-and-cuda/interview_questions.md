# Interview Questions: GPU and CUDA

These questions test your understanding of the practical aspects of using GPUs for PyTorch development.

### 1. What is the purpose of `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`? Why is this line considered best practice?

**Answer:**
The purpose of this line of code is to create a `device` object that represents the hardware you want to run your computations on.

It is considered best practice because it makes the code **portable and device-agnostic**.
-   `torch.cuda.is_available()` checks if the machine has a compatible NVIDIA GPU and if the CUDA drivers are correctly installed.
-   If it returns `True`, the `device` object is set to `'cuda'`, meaning all subsequent operations sent to this device will run on the GPU.
-   If it returns `False` (i.e., no GPU is found or it's not set up correctly), the `device` object gracefully falls back to `'cpu'`.

By writing your code to use this `device` object (e.g., `model.to(device)`, `tensor.to(device)`), you create a single script that can be run without modification on both a powerful GPU-equipped machine and a standard laptop with only a CPU. This dramatically improves the reproducibility and accessibility of your code.

### 2. You have a model on the GPU and a data tensor on the CPU. What happens when you try to execute `model(data_tensor)`? How do you fix it?

**Answer:**
This will raise a `RuntimeError`. The error message will be very specific and state that the `input` and `weight` tensors are on different devices. For example: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`.

This happens because a computational operation (like the matrix multiplication inside a linear layer) cannot be performed between tensors that reside in different memory locations (one in the GPU's VRAM and one in the system's RAM).

**How to fix it:**
You must explicitly move the data tensor to the same device as the model before the operation.
```python
# Assume model is on the GPU and data_tensor is on the CPU
# Fix:
data_tensor = data_tensor.to(device) # where device is 'cuda'
output = model(data_tensor)
```
This is the most common step to remember inside any training or evaluation loop.

### 3. Your script is running on a GPU, and you want to plot a tensor's values using `matplotlib` or convert it to a NumPy array. What must you do first and why?

**Answer:**
You must first move the tensor from the GPU back to the **CPU**.

You can do this by calling the `.cpu()` method on the tensor.
```python
# gpu_tensor is a tensor currently on the GPU
cpu_tensor = gpu_tensor.cpu()

# Now you can convert it to NumPy
numpy_array = cpu_tensor.numpy()
```

**Why is this necessary?**
Libraries like NumPy and Matplotlib are designed to work with data stored in the main system memory (RAM), which is managed by the CPU. They do not have direct access to the GPU's dedicated memory (VRAM). The `.numpy()` method specifically requires the tensor to be on the CPU. Attempting to call `.numpy()` on a GPU tensor will result in a `TypeError`. Therefore, the `.cpu()` call is the necessary intermediate step to bridge the gap between GPU-based PyTorch computations and CPU-based libraries.
