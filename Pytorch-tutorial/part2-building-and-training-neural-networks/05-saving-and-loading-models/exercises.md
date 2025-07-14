# Exercises: Saving and Loading Models

These exercises will help you practice the workflow of saving and loading model checkpoints.

## Exercise 1: Save Only the Model's Weights

**Task:** Sometimes, you only care about the model's learned parameters for inference and don't need the optimizer state or epoch number.

1.  Modify the `saving_and_loading_models_example.py` script.
2.  Instead of saving a dictionary (checkpoint), save only the `model.state_dict()` directly.
    ```python
    # Instead of torch.save(checkpoint, ...), use:
    torch.save(model.state_dict(), 'model_weights.pth')
    ```
3.  Create a new model instance and load the weights into it from this new file.

**Goal:** Understand the simplest form of model saving, which is common when deploying a model for production inference.

## Exercise 2: The Importance of `model.eval()`

**Task:** Write a script to demonstrate why `model.eval()` is important.

1.  Create a simple model that contains a `nn.Dropout` layer. A dropout layer randomly zeros some of the elements of the input tensor during training to prevent overfitting.
    ```python
    class DropoutModel(nn.Module):
        def __init__(self):
            super(DropoutModel, self).__init__()
            self.layer = nn.Linear(10, 10)
            self.dropout = nn.Dropout(p=0.5) # 50% dropout rate

        def forward(self, x):
            return self.dropout(self.layer(x))
    ```
2.  Create an instance of this model.
3.  Create a dummy input tensor of ones.
4.  Pass the input through the model twice **without** calling `model.eval()` and print the outputs. Observe if they are different.
5.  Now, call `model.eval()`.
6.  Pass the same input through the model twice again and print the outputs. Observe if they are the same.

**Goal:** See firsthand that `model.eval()` deactivates the dropout layer, leading to deterministic and consistent outputs, which is critical for inference. Without it, your model would produce random predictions each time.

## Exercise 3: Mismatched Keys Error

**Task:** Intentionally create a "mismatched keys" error when loading a `state_dict`.

1.  Define two different model architectures. For example, `ModelA` could have a layer named `layer1`, and `ModelB` could have a layer named `layer_one`.
    ```python
    class ModelA(nn.Module):
        def __init__(self):
            super(ModelA, self).__init__()
            self.layer1 = nn.Linear(10, 5)

    class ModelB(nn.Module):
        def __init__(self):
            super(ModelB, self).__init__()
            self.layer_one = nn.Linear(10, 5) # Different layer name
    ```
2.  Create an instance of `ModelA` and save its `state_dict`.
3.  Create an instance of `ModelB` and try to load the `state_dict` from `ModelA` into it.
4.  Observe the `RuntimeError` that PyTorch raises. Read the error message carefully.

**Goal:** Understand that the keys in the `state_dict` dictionary must exactly match the names of the layers in the model instance you are loading it into. This reinforces why you must have the correct model architecture defined before you can load its weights.
