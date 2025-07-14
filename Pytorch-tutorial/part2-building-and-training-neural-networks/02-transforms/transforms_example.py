# transforms_example.py

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title):
    """Helper function to display an image."""
    # Un-normalize the image
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    """
    Demonstrates the effect of various torchvision transforms on images.
    """
    # --- 1. Load the CIFAR-10 Dataset without any transforms first ---
    # We'll load it as PIL Images to apply transforms manually for visualization
    base_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, 
                                                download=True)

    # Get a single image and label from the dataset
    pil_image, label = base_dataset[0] # The first image is an airplane
    
    print(f"Original Image Type: {type(pil_image)}")
    print(f"Label: {base_dataset.classes[label]}")
    
    plt.figure(figsize=(4, 4))
    plt.imshow(pil_image)
    plt.title("Original PIL Image")
    plt.axis('off')
    plt.show()

    # --- 2. Define and Apply a Basic Transform ---
    # The most basic transform is converting the image to a tensor
    to_tensor_transform = transforms.ToTensor()
    tensor_image = to_tensor_transform(pil_image)
    
    print(f"\nImage shape after ToTensor(): {tensor_image.shape}")
    print(f"Pixel value range: {tensor_image.min()} to {tensor_image.max()}")

    # --- 3. Define a Composition of Multiple Transforms ---
    # This is a typical pipeline for data augmentation in training
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0), # Always flip for this demo
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize
    ])

    # Apply the augmentation transforms multiple times to see different results
    augmented_images = [augmentation_transforms(pil_image) for _ in range(4)]

    # Create a grid to visualize the augmented images
    img_grid = torchvision.utils.make_grid(augmented_images, nrow=4)
    imshow(img_grid, title="Data Augmentation Examples")

    # --- 4. Define Separate Transforms for Training and Validation ---
    # Training transforms include augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Validation transforms only perform necessary pre-processing
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # You would then use these in your datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                 download=True, transform=train_transforms)
    
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                               download=True, transform=val_transforms)

    print(f"\nSuccessfully created datasets with separate train and validation transforms.")
    
    # Example of getting an item from the transformed dataset
    train_sample, train_label = train_dataset[0]
    print(f"Shape of a sample from the training dataset: {train_sample.shape}")


if __name__ == '__main__':
    main()
