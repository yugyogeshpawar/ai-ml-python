# building_a_basic_neural_network_example.py

import torch
import torch.nn as nn

# --- 1. Define the Neural Network ---
class SimpleNet(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the layers of the network.
        
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output units.
        """
        # Call the constructor of the parent class (nn.Module)
        super(SimpleNet, self).__init__()
        
        # Define the layers using nn.Sequential for a clean structure
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), # First fully connected layer
            nn.ReLU(),                         # ReLU activation function
            nn.Linear(hidden_size, output_size)  # Second fully connected layer
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output of the network.
        """
        return self.layers(x)

def main():
    """
    Demonstrates how to create and use a simple neural network.
    """
    # --- 2. Set Network Parameters ---
    input_size = 10   # e.g., 10 features in the input data
    hidden_size = 32  # 32 neurons in the hidden layer
    output_size = 1   # e.g., a single regression output
    batch_size = 64   # 64 samples in a batch

    # --- 3. Create an Instance of the Network ---
    model = SimpleNet(input_size, hidden_size, output_size)
    
    # Print the model architecture
    print("--- Model Architecture ---")
    print(model)
    print("\n")

    # --- 4. Perform a Forward Pass ---
    print("--- Forward Pass ---")
    # Create a random input tensor to simulate a batch of data
    # Shape: (batch_size, input_size)
    input_tensor = torch.randn(batch_size, input_size)
    
    # Pass the input through the model
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}\n")

    # --- 5. Inspect Model Parameters ---
    print("--- Model Parameters ---")
    # nn.Module automatically tracks all learnable parameters
    # (weights and biases) of the layers defined in __init__
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

if __name__ == "__main__":
    main()
