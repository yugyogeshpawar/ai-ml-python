# Exercises: Transforms

These exercises will help you get comfortable with creating and applying data transformation pipelines.

## Exercise 1: Create a Custom Augmentation Pipeline

**Task:** Create a `transforms.Compose` pipeline that performs the following augmentations on an image, in order:
1.  Randomly rotates the image by up to 45 degrees.
2.  Randomly flips the image vertically with a probability of 30% (`p=0.3`).
3.  Converts the image to a tensor.

Load a single image from the CIFAR-10 dataset and apply this pipeline to it. Visualize the result. Since the transformations are random, try running it a few times to see different outputs.

**Goal:** Practice combining multiple random transformations into a single augmentation pipeline.

## Exercise 2: Investigate Normalization

**Task:** The `transforms.Normalize(mean, std)` transform changes the pixel values of a tensor. Let's see exactly how.

1.  Create a simple 1x1 grayscale image tensor with a pixel value of `0.5`. Its shape should be `(1, 1, 1)` for (Channels, Height, Width).
2.  Define a normalization transform with `mean=[0.5]` and `std=[0.5]`.
3.  Apply the transform to your tensor.
4.  Print the value of the tensor after normalization.

**Formula:** `output = (input - mean) / std`

Calculate the expected output by hand and verify that it matches the result from PyTorch.

**Goal:** Understand the mathematical operation behind normalization and how it affects tensor values.

## Exercise 3: Create a Test-Time Augmentation (TTA) Pipeline

**Task:** Test-Time Augmentation is a technique where you create multiple augmented versions of a test image, get a prediction for each, and average the results. This can sometimes improve performance.

Create a transform pipeline that could be used for TTA. It should create a **horizontally flipped** version of an image.

1.  Load a test image from the CIFAR-10 dataset.
2.  Create a `val_transforms` pipeline that only converts the image to a tensor and normalizes it.
3.  Create a `tta_transforms` pipeline that **horizontally flips** the image, then converts it to a tensor and normalizes it.
4.  Apply both pipelines to the same test image to get two different versions of it.
5.  Visualize both the original and the flipped test images.

**Goal:** Think about how transforms can be used creatively, not just for training data augmentation but also during inference.
