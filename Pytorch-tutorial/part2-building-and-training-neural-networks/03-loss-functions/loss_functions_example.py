# loss_functions_example.py

import torch
import torch.nn as nn

def main():
    """
    Demonstrates the usage of common PyTorch loss functions.
    """
    
    # --- 1. Mean Squared Error (MSE) for Regression ---
    print("--- 1. MSE Loss (Regression) ---")
    
    # Create an MSE loss function instance
    mse_loss = nn.MSELoss()
    
    # Imagine a model predicting house prices
    predicted_prices = torch.tensor([250000.0, 310000.0, 190000.0])
    true_prices = torch.tensor([265000.0, 300000.0, 205000.0])
    
    loss = mse_loss(predicted_prices, true_prices)
    print(f"Predicted Prices: {predicted_prices.numpy()}")
    print(f"True Prices: {true_prices.numpy()}")
    print(f"MSE Loss: {loss.item():.2f}\n")
    # The loss is the average of (250k-265k)^2, (310k-300k)^2, and (190k-205k)^2

    # --- 2. Cross-Entropy Loss for Multi-Class Classification ---
    print("--- 2. Cross-Entropy Loss (Multi-Class Classification) ---")
    
    # Create a Cross-Entropy loss function instance
    ce_loss = nn.CrossEntropyLoss()
    
    # Imagine a model classifying an image into one of 4 classes (e.g., cat, dog, bird, fish)
    # The model outputs raw scores (logits) for each class.
    # Batch size = 3 samples, Number of classes = 4
    predicted_logits = torch.randn(3, 4)
    
    # The true labels are the class indices
    true_labels = torch.tensor([1, 0, 3]) # 1=dog, 0=cat, 3=fish
    
    loss = ce_loss(predicted_logits, true_labels)
    print(f"Predicted Logits (shape {predicted_logits.shape}):\n{predicted_logits}")
    print(f"True Labels (shape {true_labels.shape}): {true_labels}")
    print(f"Cross-Entropy Loss: {loss.item():.4f}\n")
    # Note: We did NOT apply Softmax to the logits. CrossEntropyLoss does it internally.

    # --- 3. Binary Cross-Entropy with Logits Loss for Binary Classification ---
    print("--- 3. BCE With Logits Loss (Binary Classification) ---")
    
    # This version is numerically more stable than using a Sigmoid layer + BCELoss separately.
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    
    # Imagine a model predicting if an email is spam (1) or not spam (0)
    # The model outputs a single raw logit for each sample.
    # Batch size = 4 samples
    predicted_logits_binary = torch.randn(4)
    
    # The true labels are floats (0.0 or 1.0)
    true_labels_binary = torch.tensor([1.0, 0.0, 1.0, 0.0]) # spam, not spam, spam, not spam
    
    loss = bce_with_logits_loss(predicted_logits_binary, true_labels_binary)
    print(f"Predicted Logits: {predicted_logits_binary.numpy()}")
    print(f"True Labels: {true_labels_binary.numpy()}")
    print(f"BCE With Logits Loss: {loss.item():.4f}\n")
    # Note: We did NOT apply Sigmoid. BCEWithLogitsLoss does it internally.

if __name__ == '__main__':
    main()
