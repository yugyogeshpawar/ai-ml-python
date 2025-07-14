# Part 2 Project: Image Classification with a CNN on CIFAR-10

This project integrates all the concepts from Part 2 to build, train, and evaluate a complete image classification model. We will use a Convolutional Neural Network (CNN) to classify images from the popular CIFAR-10 dataset.

## Goal

The goal is to build a robust image classifier that can correctly identify the class of an image from one of the 10 categories in the CIFAR-10 dataset. This project will demonstrate the full, end-to-end workflow of a typical computer vision task in PyTorch.

## The CIFAR-10 Dataset

CIFAR-10 is a widely used benchmark dataset in computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:
`'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'`

## Project Steps

The `cifar10_cnn.py` script is structured to follow a standard deep learning workflow:

1.  **Setup:**
    -   Set the device to GPU (`cuda`) if available, otherwise CPU.
    -   Define hyperparameters like learning rate, batch size, and number of epochs.

2.  **Data Preparation:**
    -   Define separate `transforms` for the training set (with data augmentation like `RandomCrop` and `RandomHorizontalFlip`) and the test set (with only necessary pre-processing).
    -   Use `torchvision.datasets.CIFAR10` to load the training and test data.
    -   Create `DataLoader` instances for both training and testing to handle batching and shuffling.

3.  **Define the CNN Model:**
    -   A `ConvNet` class is defined, inheriting from `nn.Module`.
    -   The architecture consists of two convolutional blocks followed by three fully connected layers.
    -   Each convolutional block has a `Conv2d` layer, a `ReLU` activation, and a `MaxPool2d` layer to downsample the feature maps.
    -   The output of the convolutional layers is flattened before being passed to the fully connected layers.

4.  **Instantiate Model, Loss, and Optimizer:**
    -   An instance of the `ConvNet` is created and moved to the target device.
    -   The loss function is `nn.CrossEntropyLoss`, as this is a multi-class classification problem.
    -   The optimizer is `optim.Adam`, a robust choice for this kind of task.

5.  **Training Loop:**
    -   The script iterates for a specified number of epochs.
    -   Inside each epoch, it iterates through the `train_loader`.
    -   For each batch, it performs the standard training steps: forward pass, loss calculation, `zero_grad()`, `backward()`, and `step()`.

6.  **Evaluation Loop:**
    -   After training, the model's performance is evaluated on the test set.
    -   Crucially, the model is set to evaluation mode using `model.eval()`.
    -   The evaluation loop iterates through the `test_loader`.
    -   It calculates the accuracy by comparing the model's predictions (`torch.max`) with the true labels.
    -   No gradients are computed (`with torch.no_grad():`) to save memory and computation.

7.  **Save the Model:**
    -   Finally, the trained model's `state_dict` is saved to a file (`cifar10_cnn.pth`) for future use.

This project serves as a complete template for tackling image classification problems in PyTorch.
