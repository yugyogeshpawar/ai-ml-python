# pytorch_tensors_example.py

import torch
import numpy as np

def main():
    """
    Demonstrates the creation, attributes, and operations of PyTorch Tensors.
    """
    # --- 1. Creating Tensors ---
    print("--- 1. Creating Tensors ---")
    
    # From a Python list
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"Tensor from list:\n {x_data}\n")

    # From a NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"Tensor from NumPy array:\n {x_np}\n")

    # Tensors with specific shapes and values
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor:\n {rand_tensor}\n")
    print(f"Ones Tensor:\n {ones_tensor}\n")
    print(f"Zeros Tensor:\n {zeros_tensor}\n")

    # --- 2. Tensor Attributes ---
    print("--- 2. Tensor Attributes ---")
    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}\n")

    # --- 3. Tensor Operations ---
    print("--- 3. Tensor Operations ---")
    
    # Indexing and slicing
    tensor = torch.ones(4, 4)
    print(f"Original tensor:\n {tensor}")
    tensor[:, 1] = 0  # Set all values in the second column to 0
    print(f"Tensor after slicing:\n {tensor}\n")

    # Joining tensors
    t1 = torch.cat([tensor, tensor], dim=1)
    print(f"Concatenated tensor shape: {t1.shape}\n")

    # Arithmetic operations
    # Element-wise multiplication
    print(f"Element-wise multiplication:\n {tensor.mul(tensor)}\n")
    # Matrix multiplication
    print(f"Matrix multiplication:\n {tensor.matmul(tensor.T)}\n")

    # In-place operations
    print(f"Original tensor before in-place add:\n {tensor}")
    tensor.add_(5)
    print(f"Tensor after in-place add:\n {tensor}\n")

    # --- 4. NumPy Bridge ---
    print("--- 4. NumPy Bridge ---")
    
    # Tensor to NumPy
    t = torch.ones(5)
    n = t.numpy()
    print(f"Original tensor: {t}")
    print(f"Converted NumPy array: {n}")
    t.add_(1)
    print(f"Tensor after modification: {t}")
    print(f"NumPy array after tensor modification: {n}\n")

    # NumPy to Tensor
    n = np.ones(5)
    t = torch.from_numpy(n)
    print(f"Original NumPy array: {n}")
    print(f"Converted tensor: {t}")
    np.add(n, 1, out=n)
    print(f"NumPy array after modification: {n}")
    print(f"Tensor after NumPy modification: {t}")

if __name__ == "__main__":
    main()
