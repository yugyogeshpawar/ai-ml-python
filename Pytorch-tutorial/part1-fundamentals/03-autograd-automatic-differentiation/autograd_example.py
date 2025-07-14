# autograd_example.py

import torch

def main():
    """
    Demonstrates the core functionality of PyTorch's autograd engine.
    """
    # --- 1. Basic Gradient Calculation ---
    print("--- 1. Basic Gradient Calculation ---")
    
    # Create tensors for a simple linear equation: y = w * x + b
    # We want to find the gradients of a loss function with respect to w and b.
    x = torch.tensor(2.0)
    y_true = torch.tensor(10.0)
    
    # Set requires_grad=True to track operations on these tensors
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    print(f"Initial parameters: w = {w.item():.3f}, b = {b.item():.3f}")

    # Forward pass: compute the predicted y and the loss
    y_pred = w * x + b
    loss = (y_pred - y_true) ** 2  # A simple squared error loss
    
    print(f"Predicted y: {y_pred.item():.3f}")
    print(f"Loss: {loss.item():.3f}\n")

    # Backward pass: compute the gradients
    # This calculates d(loss)/dw and d(loss)/db
    loss.backward()

    # The gradients are stored in the .grad attribute
    print(f"Gradient of loss w.r.t. w: {w.grad.item():.3f}")
    print(f"Gradient of loss w.r.t. b: {b.grad.item():.3f}\n")

    # --- 2. Disabling Gradient Tracking ---
    print("--- 2. Disabling Gradient Tracking ---")
    
    print(f"w.requires_grad before no_grad block: {w.requires_grad}")
    
    # Use torch.no_grad() for operations that should not be tracked
    with torch.no_grad():
        # This computation will not be part of the computation graph
        y_inference = w * x + b
        print(f"y_inference.requires_grad inside no_grad block: {y_inference.requires_grad}")
    
    print(f"w.requires_grad after no_grad block: {w.requires_grad}\n")

    # Using .detach() to get a new tensor that is not part of the graph
    detached_w = w.detach()
    print(f"Original w.requires_grad: {w.requires_grad}")
    print(f"Detached w.requires_grad: {detached_w.requires_grad}\n")

    # --- 3. Gradient Accumulation ---
    print("--- 3. Gradient Accumulation ---")
    
    # Gradients are accumulated by default. You need to zero them out
    # in a training loop before each new backward pass.
    
    # Let's calculate another loss and backpropagate
    new_loss = (w * x + b - y_true) ** 2
    new_loss.backward()
    
    print(f"Gradient of w after first backward pass: {w.grad.item():.3f}")
    
    # If we call backward() again, the new gradients are ADDED to the old ones
    another_loss = (w * x + b - y_true) ** 2
    another_loss.backward()
    
    print(f"Gradient of w after second backward pass (accumulated): {w.grad.item():.3f}")

    # In a real training loop, you would zero the gradients
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
        
    print(f"Gradient of w after zeroing: {w.grad.item():.3f}")

if __name__ == "__main__":
    main()
