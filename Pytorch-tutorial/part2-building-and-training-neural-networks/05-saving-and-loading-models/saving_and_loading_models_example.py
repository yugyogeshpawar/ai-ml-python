# saving_and_loading_models_example.py

import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Define a Simple Model ---
# We'll use a very simple linear model for this demonstration.
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def main():
    """
    Demonstrates saving and loading a model checkpoint.
    """
    # --- Setup ---
    input_size = 10
    output_size = 1
    learning_rate = 0.01
    model_save_path = 'simple_model_checkpoint.pth'

    # --- 2. Create Model and Optimizer ---
    model = SimpleModel(input_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("--- Initial Model ---")
    # Print the initial, randomly initialized weight
    print("Initial weight:", model.linear.weight.data)
    
    # --- 3. Simulate a Training Step and Save the Model ---
    print("\n--- Saving Model ---")
    
    # Create some dummy data
    dummy_input = torch.randn(1, input_size)
    dummy_target = torch.randn(1, output_size)
    
    # Simulate one training step
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    print("Weight after one training step:", model.linear.weight.data)

    # Create a checkpoint dictionary
    checkpoint = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }
    
    # Save the checkpoint
    torch.save(checkpoint, model_save_path)
    print(f"Model checkpoint saved to '{model_save_path}'")

    # --- 4. Load the Model ---
    print("\n--- Loading Model ---")
    
    # To load the model, we first need to create a new instance of the model architecture.
    # This new model will have fresh, randomly initialized weights.
    new_model = SimpleModel(input_size, output_size)
    new_optimizer = optim.SGD(new_model.parameters(), lr=learning_rate)
    
    print("Weight of new_model before loading:", new_model.linear.weight.data)
    
    # Load the checkpoint from the file
    loaded_checkpoint = torch.load(model_save_path)
    
    # Load the state dictionaries into the new model and optimizer
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    
    # You can also retrieve other saved information
    start_epoch = loaded_checkpoint['epoch']
    last_loss = loaded_checkpoint['loss']
    
    print("Weight of new_model after loading:", new_model.linear.weight.data)
    print(f"Model loaded successfully. Ready to resume training from epoch {start_epoch + 1}.")

    # --- 5. Using the Loaded Model for Inference ---
    print("\n--- Inference with Loaded Model ---")
    
    # IMPORTANT: Always call model.eval() before inference to set layers
    # like dropout and batch norm to evaluation mode.
    new_model.eval()
    
    # Use torch.no_grad() to disable gradient calculations for inference,
    # which saves memory and computation.
    with torch.no_grad():
        # Create some test data
        test_input = torch.randn(1, input_size)
        
        # Get the prediction from the loaded model
        prediction = new_model(test_input)
        print(f"Input: {test_input.numpy()}")
        print(f"Prediction: {prediction.item():.4f}")

if __name__ == '__main__':
    main()
