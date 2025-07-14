# Exercises: PyTorch Tensors

These exercises will help you practice creating and manipulating PyTorch tensors.

## Exercise 1: Create Tensors of Different Data Types

**Task:** Create three different tensors with the same shape `(3, 3)` but with different data types:
1.  A tensor of floating-point numbers (`torch.float32`).
2.  A tensor of integers (`torch.int32`).
3.  A tensor of booleans (`torch.bool`).

Create them using `torch.ones()` and specify the `dtype` argument. Print each tensor and its `dtype` attribute.

**Goal:** Understand how to control the data type of a tensor, which is important for managing memory and precision in your models.

## Exercise 2: Tensor Concatenation

**Task:** Create two tensors, `A` and `B`, both with shape `(2, 3)`.
1.  Concatenate them along dimension 0 (stacking them vertically). The resulting tensor should have a shape of `(4, 3)`.
2.  Concatenate them along dimension 1 (stacking them horizontally). The resulting tensor should have a shape of `(2, 6)`.

Print the shape of the resulting tensor after each operation.

**Goal:** Practice using `torch.cat` to combine tensors, a common operation when preparing data batches or manipulating feature maps in neural networks.

## Exercise 3: The NumPy Bridge and Memory Sharing

**Task:** Demonstrate that a CPU tensor and a NumPy array can share the same memory.

1.  Create a 1D PyTorch tensor of zeros with 5 elements.
2.  Convert it to a NumPy array using the `.numpy()` method.
3.  Using NumPy, change the value of the element at index 2 of the NumPy array to `100`.
4.  Print the original PyTorch tensor.

**Goal:** Verify that modifying the NumPy array also modifies the source PyTorch tensor. This will prove they share the same underlying memory. What do you expect to see?
