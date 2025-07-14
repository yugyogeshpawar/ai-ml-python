# datasets_and_dataloaders_example.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    """
    Demonstrates how to use torchvision Datasets and DataLoaders.
    """
    # --- 1. Define Transformations ---
    # Transformations are applied to the data as it's loaded.
    # We'll convert images to tensors and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to a FloatTensor.
        transforms.Normalize((0.5,), (0.5,)) # Normalize with mean and standard deviation.
    ])

    # --- 2. Load the Dataset ---
    # We'll use the built-in MNIST dataset from torchvision.
    # `download=True` will download the data if it's not found in the `root` directory.
    
    # Load the training data
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                               train=True, 
                                               transform=transform,  
                                               download=True)
    
    # Load the test data
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=False, 
                                              transform=transform)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # --- 3. Create the DataLoader ---
    # The DataLoader wraps the Dataset and provides an iterator for easy batching.
    batch_size = 64

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, # Shuffle data for training
                              num_workers=2) # Use 2 subprocesses for data loading

    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, # No need to shuffle test data
                             num_workers=2)

    # --- 4. Iterate and Inspect a Batch ---
    print("\n--- Inspecting a Batch from the DataLoader ---")
    
    # Get one batch of training images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Print the shape of the batch
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    
    # The images batch shape is (batch_size, channels, height, width).
    # For MNIST, it's (64, 1, 28, 28) because the images are grayscale (1 channel).

    # --- 5. Visualize a Few Images ---
    print("\n--- Visualizing a Few Images ---")
    
    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images[:16], nrow=4)
    
    # Un-normalize the images to display them correctly
    img_grid = img_grid / 2 + 0.5 
    
    # Convert to NumPy for plotting
    np_img = img_grid.numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title("Sample MNIST Images")
    plt.axis('off')
    plt.show()

    print("Corresponding labels for the first row:")
    print(labels[:4].numpy())

if __name__ == '__main__':
    main()
