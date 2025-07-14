# Part 1, Topic 2: PyTorch Tensors

In the previous lesson, you learned what PyTorch is and set up your environment. Now, let's dive into the most fundamental building block of PyTorch: the **Tensor**.

## What is a `torch.Tensor`?

A `torch.Tensor` is a multi-dimensional array, similar to a NumPy `ndarray`. It is the primary data structure used in PyTorch to store and manipulate data. Tensors can represent scalars (0-dimensional), vectors (1-dimensional), matrices (2-dimensional), or higher-dimensional arrays.

What makes PyTorch Tensors special for deep learning is their ability to:
1.  Run on **GPUs** for massive computational speedups.
2.  Keep track of the **computation graph** for automatic differentiation with `autograd`.

## Creating Tensors

You can create tensors in several ways.

### 1. From Existing Data

You can create a tensor directly from Python lists or NumPy arrays.

```python
import torch
import numpy as np

# From a Python list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
# tensor([[1, 2],
#         [3, 4]])

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
# tensor([[1, 2],
#         [3, 4]])
```

### 2. From Another Tensor's Properties

You can create a new tensor that has the same properties (like shape and data type) as another tensor.

```python
# Create a tensor of ones with the same shape as x_data
x_ones = torch.ones_like(x_data)
print(x_ones)
# tensor([[1, 1],
#         [1, 1]])

# Create a random tensor with the same shape
x_rand = torch.rand_like(x_data, dtype=torch.float) # Override datatype
print(x_rand)
# tensor([[0.6206, 0.3065],
#         [0.4125, 0.8486]])
```

### 3. With Random or Constant Values

You can create tensors with a specific shape, filled with random or constant values.

```python
shape = (2, 3,) # A tuple defining the shape
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

## Tensor Attributes

Every tensor has attributes that describe its `shape`, `dtype` (data type), and the `device` where it is stored (CPU or GPU).

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

## Tensor Operations

PyTorch provides a rich library of over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation, and more.

### 1. Standard Indexing and Slicing

Tensors can be indexed and sliced just like NumPy arrays.

```python
tensor = torch.ones(4, 4)
tensor[:, 1] = 0 # Set all values in the second column to 0
print(tensor)
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
```

### 2. Joining Tensors

You can join a sequence of tensors along a given dimension using `torch.cat`.

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1.shape)
# torch.Size([4, 12])
```

### 3. Arithmetic Operations

```python
# Element-wise multiplication
print(tensor.mul(tensor))
# Or equivalently:
print(tensor * tensor)

# Matrix multiplication
print(tensor.matmul(tensor.T)) # .T transposes the tensor
# Or equivalently:
print(tensor @ tensor.T)
```

### 4. In-place Operations

Operations that modify a tensor in-place are post-fixed with an `_`. For example, `x.copy_(y)` or `x.t_()` will change `x`.

```python
tensor.add_(5) # Adds 5 to each element of the tensor in-place
print(tensor)
```

**Note:** In-place operations save some memory but can be problematic when computing derivatives because they immediately destroy the original value. Therefore, their use is discouraged in most cases.

## NumPy Bridge

Tensors on the CPU and NumPy arrays can share their underlying memory locations. Changing the NumPy array will change the PyTorch tensor, and vice-versa.

### Tensor to NumPy

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # Convert tensor to NumPy array
print(f"n: {n}")

t.add_(1) # Modify the tensor in-place
print(f"t after modification: {t}")
print(f"n after tensor modification: {n}") # NumPy array also changed
```

### NumPy to Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n) # Convert NumPy array to tensor

np.add(n, 1, out=n) # Modify the NumPy array in-place
print(f"n after modification: {n}")
print(f"t after numpy modification: {t}") # Tensor also changed
```

This seamless interoperability makes it easy to leverage the vast Python data science ecosystem.

## Step-by-Step Code Tutorial

The `pytorch_tensors_example.py` script demonstrates these concepts in a runnable format. It covers creating tensors, performing operations, and the NumPy bridge.
