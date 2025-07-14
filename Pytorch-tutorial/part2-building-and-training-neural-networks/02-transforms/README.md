# Part 2, Topic 2: Transforms

When working with real-world data, especially images, you rarely feed the raw data directly into a neural network. You need to perform a series of pre-processing steps to get the data into the right format and to augment it for better model performance. In PyTorch, this is handled by **Transforms**.

## What are Transforms?

Transforms are common data transformations available in the `torchvision.transforms` module. They can be chained together using `transforms.Compose` to create a pipeline of pre-processing steps.

These transformations are typically applied in two places:
1.  **Pre-processing:** To convert data into a format suitable for the model (e.g., converting a PIL Image to a tensor, normalizing pixel values).
2.  **Data Augmentation:** To artificially increase the diversity of the training dataset by applying random transformations (e.g., random rotations, flips, or crops).

## Why Use Transforms?

1.  **Data Consistency:** Transforms ensure that all data fed to the model has a consistent format (e.g., same size, same normalization). This is crucial for stable training.
2.  **Data Augmentation:** By applying random transformations to the training data, you create new, slightly modified versions of existing images. This helps the model become more robust and generalize better to unseen data, reducing overfitting.

**Important:** Data augmentation should only be applied to the **training set**. The validation and test sets should only undergo the necessary pre-processing (like resizing and normalization) to ensure that you are evaluating your model on a consistent, unmodified version of the data.

## Common Transforms

The `torchvision.transforms` module provides a wide variety of transformations. Here are some of the most common ones for image data:

### Conversion and Resizing
-   `transforms.ToTensor()`: Converts a `PIL.Image` or `numpy.ndarray` (with shape H x W x C) into a `torch.FloatTensor` (with shape C x H x W) and scales the pixel values from the range `[0, 255]` to `[0.0, 1.0]`. This is almost always the first step.
-   `transforms.ToPILImage()`: Converts a tensor or ndarray into a PIL Image.
-   `transforms.Resize((h, w))`: Resizes the input image to a given size.
-   `transforms.CenterCrop(size)`: Crops the central portion of the image to the given size.

### Normalization
-   `transforms.Normalize(mean, std)`: Normalizes a tensor image with a mean and standard deviation. For an `n`-channel image, `mean` and `std` should be sequences of `n` values. The formula is `output[channel] = (input[channel] - mean[channel]) / std[channel]`. Normalizing data to have a zero mean and unit variance can help the model train faster and more stably.

### Data Augmentation (Random Transforms)
-   `transforms.RandomHorizontalFlip(p=0.5)`: Horizontally flips the given image randomly with a given probability `p` (default is 0.5).
-   `transforms.RandomRotation(degrees)`: Rotates the image by a random angle selected from `(-degrees, +degrees)`.
-   `transforms.RandomResizedCrop(size)`: Crops a random portion of an image and resizes it to a given size. This is a very common augmentation technique.
-   `transforms.ColorJitter()`: Randomly changes the brightness, contrast, saturation, and hue of an image.

## Composing Transforms

The real power of transforms comes from chaining them together with `transforms.Compose`. This creates a single pipeline that applies a series of transformations in order.

### Example: Training and Validation Transforms

Here's a typical example of how you would define separate transform pipelines for your training and validation sets.

```python
import torchvision.transforms as transforms

# For training: include data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# For validation/testing: only include necessary pre-processing
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```
*(Note: The mean and std values above are the standard values for models pre-trained on the ImageNet dataset).*

You would then pass these transform pipelines to your `Dataset` objects:

```python
train_dataset = MyDataset(..., transform=train_transforms)
val_dataset = MyDataset(..., transform=val_transforms)
```

The `transforms_example.py` script shows how to apply these transformations and visualize their effects on an image.
