# optimizers_example.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. Define a Simple CNN Model ---
# We'll create a simple Convolutional Neural Network for the MNIST dataset.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # The input features to the linear layer depend on the output of the conv layers
        # For MNIST 28x28 -> pool -> 14x14 -> pool -> 7x7. So, 32 channels * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 10) # 10 output classes for MNIST

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 32 * 7 * 7) # Flatten the tensor
        out = self.fc1(out)
        return out

def main():
    """
    Demonstrates the full training loop including the optimizer.
    """
    # --- 2. Setup: Device, Hyperparameters, Data ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 2 # Set to a small number for a quick demo

    # Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std for MNIST
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Initialize Model, Loss Function, and Optimizer ---
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. The Training Loop ---
    print("\n--- Starting Training ---")
    model.train() # Set the model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # a. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # b. Backward pass and optimization
            optimizer.zero_grad() # Clear gradients from previous iteration
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights

            total_loss += loss.item()

            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f"--- End of Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f} ---")

    print("--- Training Finished ---")

    # --- 5. (Optional) A simple check after training ---
    # Let's see the initial weights of the first layer
    initial_weights = model.conv1.weight.clone()
    
    # After optimizer.step(), the weights should have changed.
    # We can verify this by comparing the weights before and after one step.
    # This is just a sanity check to see the optimizer is working.
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    final_weights = model.conv1.weight
    
    # The weights should not be equal anymore
    are_weights_equal = torch.equal(initial_weights, final_weights)
    print(f"\nAre conv1 weights the same after one optimization step? {are_weights_equal}")


if __name__ == '__main__':
    main()
