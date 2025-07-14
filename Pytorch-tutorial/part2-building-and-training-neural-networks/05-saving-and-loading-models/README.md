# Part 2, Topic 5: Saving and Loading Models

Training a deep learning model can take hours, days, or even weeks. It would be impractical if you had to retrain your model from scratch every time you wanted to use it. That's why learning how to save and load your model's progress is an essential skill.

In PyTorch, this is primarily done by saving the model's **state dictionary**.

## What is a `state_dict`?

A model's `state_dict` is a Python dictionary object that maps each layer to its learnable parameters (weights and biases). It contains all the "state" that your model has learned.

-   For a model, the `state_dict` contains the weights and biases of all its layers.
-   For an optimizer, the `state_dict` contains its internal state and the hyperparameters it uses.

Saving the `state_dict` is the recommended approach because it is flexible and lightweight. It decouples the model's state from the code that defines the model's architecture.

## How to Save a Model

The process involves two main steps:
1.  Get the model's state dictionary using `model.state_dict()`.
2.  Save this dictionary to a file using `torch.save()`. The common convention is to use a `.pt` or `.pth` file extension.

```python
# Assume 'model' is your trained nn.Module instance
# and 'optimizer' is your optimizer instance

# It's good practice to save more than just the model's state.
# Saving the epoch, loss, and optimizer state is useful for resuming training.
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

# Save the checkpoint to a file
torch.save(checkpoint, 'model_checkpoint.pth')
```

This dictionary, often called a **checkpoint**, contains a snapshot of your training process, making it easy to resume later.

## How to Load a Model

Loading a model involves the reverse process:
1.  First, you must create an instance of your model architecture. **PyTorch does not save the model's code**, only its parameters.
2.  Load the saved `state_dict` from the file using `torch.load()`.
3.  Load this dictionary into your model instance using `model.load_state_dict()`.

```python
# 1. Create an instance of the model architecture
model = MyModelClass(*args, **kwargs)
optimizer = MyOptimizerClass(model.parameters(), lr=...)

# 2. Load the checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# 3. Load the state into the model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Model loaded from epoch {epoch} with loss {loss}")
```

### Important: `model.eval()`

After loading a trained model for **inference** (i.e., to make predictions, not to continue training), you must always call `model.eval()`.

`model.eval()` sets the model to evaluation mode. This is important because some layers, like **Dropout** and **BatchNorm**, behave differently during training and evaluation.
-   **Dropout layers** are turned off during evaluation.
-   **BatchNorm layers** use their running statistics instead of batch statistics.

Forgetting to call `model.eval()` can lead to inconsistent and incorrect predictions.

If you want to resume training after loading a model, you would call `model.train()` instead to ensure these layers are in their correct training mode.

## Saving the Entire Model

It is also possible to save the entire model object, including its architecture, using `torch.save(model, 'model.pth')`.

However, this method is **not recommended**.
-   **Less Flexible:** The serialized data is tightly coupled to the specific classes and directory structure used when the model was saved. If you refactor your code or move your files, you might not be able to load the model.
-   **Less Portable:** It can break when used in other projects or after code refactoring.

Sticking to saving the `state_dict` is the most robust and standard practice in the PyTorch community.

The `saving_and_loading_models_example.py` script provides a complete, runnable example of saving a model's checkpoint and loading it back for inference.
