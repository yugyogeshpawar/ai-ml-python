# Exercises: Datasets and DataLoaders

These exercises will help you practice using `Dataset` and `DataLoader`.

## Exercise 1: Explore a Different Dataset

**Task:** The `torchvision.datasets` module contains several other built-in datasets, such as `CIFAR10`. Your task is to load the CIFAR-10 training dataset.

1.  Import `torchvision`.
2.  Use `torchvision.datasets.CIFAR10` to create a `Dataset` object for the **training** set.
3.  Make sure to set `download=True` and apply a `transforms.ToTensor()` transformation.
4.  Print the total number of samples in the dataset.

**Goal:** Become familiar with loading different types of datasets from `torchvision`.

## Exercise 2: Change DataLoader Parameters

**Task:** Create a `DataLoader` for the CIFAR-10 training dataset with the following specifications:
-   A `batch_size` of 128.
-   `shuffle` set to `True`.
-   `num_workers` set to 2.

Then, iterate through the `DataLoader` to get one batch and print the shape of the images tensor and the labels tensor from that batch.

**Goal:** Practice configuring a `DataLoader` with different parameters to see how it affects the output batch.

## Exercise 3: Create a Custom Dataset for Text

**Task:** Imagine you have a simple text dataset represented as a list of sentences and a corresponding list of labels. Create a custom `Dataset` for this data.

```python
# Sample data
sentences = [
    "i love pytorch",
    "pytorch is easy",
    "i hate numpy",
    "numpy is hard"
]
labels = [1, 1, 0, 0] # 1 for positive, 0 for negative
```

1.  Create a class `TextDataset` that inherits from `torch.utils.data.Dataset`.
2.  In the `__init__` method, store the sentences and labels.
3.  Implement the `__len__` method to return the number of sentences.
4.  Implement the `__getitem__` method to return the sentence and its corresponding label at a given index.

Instantiate your `TextDataset` and use a `DataLoader` to fetch the first batch (e.g., with a `batch_size` of 2). Print the contents of the batch.

**Goal:** Understand that `Dataset` is a flexible class that can be adapted for any type of data, not just images. This exercise demonstrates its use for a simple text classification scenario. (Note: In a real NLP task, you would also perform tokenization and numericalization in `__getitem__`).
