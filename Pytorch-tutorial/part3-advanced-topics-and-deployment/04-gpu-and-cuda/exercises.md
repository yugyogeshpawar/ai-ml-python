# Exercises: GPU and CUDA

These exercises will help you practice the device-placement workflow in PyTorch.

## Exercise 1: The Device Placement Golden Rule

**Task:** Answer the following question: What is the single most important rule to remember when performing an operation between two or more tensors (including a model and an input tensor)? What happens if you violate this rule?

**Goal:** Solidify the most fundamental concept of GPU computing in PyTorch.

## Exercise 2: Modify a Training Loop for GPU

**Task:** Below is a simplified training loop that is written only for the CPU. Your task is to modify it so that it will run on a GPU if one is available.

```python
# Original CPU-only code
import torch
import torch.nn as nn

# Dummy model and data
model = nn.Linear(10, 2)
dataloader = [(torch.randn(64, 10), torch.randint(0, 2, (64,))) for _ in range(10)]

# --- MODIFICATION REQUIRED BELOW ---

for data, labels in dataloader:
    # Forward pass
    outputs = model(data)
    
    # Dummy loss calculation
    loss = outputs.sum()
    
    # Backward pass
    loss.backward()
```

1.  Add the code to define the `device` object.
2.  Add the line to move the `model` to the `device`.
3.  Modify the loop to move the `data` and `labels` tensors to the `device`.

**Goal:** Practice the standard pattern for making a PyTorch training script device-agnostic.

## Exercise 3: Debugging a Device Mismatch Error

**Task:** The following code snippet will produce a `RuntimeError`. Read the code, predict what the error message will say, and then write the single line of code that fixes it.

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(10, 2).to(device)
input_tensor = torch.randn(4, 10) # This tensor is on the CPU

# This line will cause an error
output = model(input_tensor) 
```

1.  **Predict the error:** What will the error message be about?
2.  **Fix the code:** Add the one line needed to make the `model(input_tensor)` call succeed.

**Goal:** Learn to recognize and quickly fix the most common error encountered when using GPUs in PyTorch.
