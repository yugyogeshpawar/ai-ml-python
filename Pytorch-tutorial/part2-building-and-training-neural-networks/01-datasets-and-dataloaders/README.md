# Part 2, Topic 1: Datasets and DataLoaders

In Part 1, we worked with simple, manually created tensors. In real-world machine learning, data is rarely that clean or simple. You need an efficient way to load, batch, and pre-process large datasets. PyTorch provides two powerful classes for this: `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.

## The Problem with Manual Data Handling

Imagine you have a dataset of 100,000 images. You can't load them all into memory at once. You also need to:
-   Load data from disk efficiently.
-   Apply pre-processing transformations (e.g., resizing, normalization).
-   Shuffle the data for training to prevent the model from learning the order of the data.
-   Group the data into batches for training.

Doing this manually is complex and error-prone. `Dataset` and `DataLoader` abstract this complexity away.

## `torch.utils.data.Dataset`

A `Dataset` is an abstract class that represents a dataset. To create a custom dataset in PyTorch, you need to create a class that inherits from `Dataset` and implements three methods:

1.  **`__init__(self)`**: This is where you perform initial setup, such as loading file paths or reading a CSV file with labels. You don't load the actual data (like images) here, only the metadata.
2.  **`__len__(self)`**: This method must return the total number of samples in the dataset. `DataLoader` uses this to know how many samples to expect.
3.  **`__getitem__(self, idx)`**: This method is responsible for loading and returning a single sample from the dataset at a given index `idx`. This is where you would read an image from a file, apply transformations, and return the sample (e.g., an image tensor and its corresponding label).

### Example: A Custom Image Dataset

Let's imagine a dataset of images stored in a directory, with labels in a CSV file.

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

## `torch.utils.data.DataLoader`

The `DataLoader` is a wrapper around a `Dataset`. It takes a `Dataset` object and provides an iterator that you can loop over to get batches of data. It handles all the complexity of batching, shuffling, and parallel data loading.

### Key `DataLoader` Arguments

-   `dataset`: The `Dataset` object to wrap.
-   `batch_size`: The number of samples per batch.
-   `shuffle`: If `True`, the data is reshuffled at every epoch. This is crucial for training to ensure the model doesn't learn from the order of the data. It should be `False` for validation and testing.
-   `num_workers`: The number of subprocesses to use for data loading. A value greater than 0 enables multi-process data loading, which can significantly speed up the process by loading data in the background while the GPU is busy with training.

### Using the `DataLoader`

Once you have a `Dataset`, creating and using a `DataLoader` is simple.

```python
# Create an instance of our custom dataset
my_dataset = CustomImageDataset(...)

# Create the DataLoader
train_loader = torch.utils.data.DataLoader(dataset=my_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=4)

# Now you can iterate over it in your training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images is a tensor of shape (64, C, H, W)
        # labels is a tensor of shape (64)
        
        # Move data to the GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Your training code here...
```

## Summary

-   **`Dataset`**: An object that knows how to access individual data samples and their labels. It abstracts away the logic of finding and loading one piece of data.
-   **`DataLoader`**: An iterator that takes a `Dataset` and automatically provides batches of shuffled data, often using multiple workers for efficiency.

Together, they form a powerful and efficient pipeline for feeding data to your PyTorch models. The `datasets_and_dataloaders_example.py` script provides a runnable example using a built-in dataset from `torchvision`.
