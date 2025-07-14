# Part 3, Topic 1: Transfer Learning

Training a state-of-the-art deep learning model from scratch requires a massive amount of data and computational resources (often, weeks of training on multiple GPUs). For most practical applications, this is not feasible.

This is where **transfer learning** comes in. Transfer learning is a technique where you take a model that has been pre-trained on a very large dataset (like ImageNet, which has over 14 million images) and adapt it for your own, usually much smaller, dataset.

## Why Use Transfer Learning?

The core idea is that a model trained on a large and diverse dataset learns general features that are useful for many different tasks. For example, a model trained on ImageNet learns to recognize low-level features like edges, textures, and shapes, as well as higher-level features like eyes, wheels, and faces.

By using a pre-trained model, you can:
1.  **Save Massive Amounts of Time and Resources:** You don't need to train a model from scratch.
2.  **Achieve Higher Performance:** Pre-trained models provide a much better starting point than random initialization, often leading to higher accuracy, especially when you have limited data.
3.  **Require Less Data:** Since the model has already learned general features, you can achieve good results with a much smaller dataset than would be required to train a model from the ground up.

## `torchvision.models`

PyTorch makes it incredibly easy to use pre-trained models through the `torchvision.models` module. This module provides instant access to dozens of state-of-the-art model architectures, such as:
-   **ResNet** (Residual Networks)
-   **VGG**
-   **Inception**
-   **MobileNet**
-   **EfficientNet**

You can load these models with a single line of code, and you can choose whether to get a model with random weights or one that has been **pre-trained on ImageNet**.

```python
import torchvision.models as models

# Load a pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)
```

## Fine-Tuning a Pre-trained Model

There are two common strategies for using a pre-trained model:

### 1. Feature Extraction

In this approach, you **freeze the weights** of the entire pre-trained model except for the final classification layer. You then replace the final layer with a new one that is tailored to your specific number of classes.

You only train the weights of this new, final layer. This is a fast and effective approach when your dataset is small and similar to the original dataset the model was trained on.

**Steps:**
1.  Load the pre-trained model.
2.  Freeze all the parameters in the model by setting `param.requires_grad = False`.
3.  Replace the final layer (the "classifier" or "head") with a new, unfrozen layer that has the correct number of output units for your task.
4.  Train the model. Only the weights of the new final layer will be updated.

```python
model = models.resnet18(pretrained=True)

# Freeze all the parameters
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the classifier
num_ftrs = model.fc.in_features

# Replace the final layer with a new one
model.fc = nn.Linear(num_ftrs, num_classes) # num_classes is the number of classes in your dataset

# Now, only the parameters of model.fc have requires_grad=True and will be trained.
```

### 2. Fine-Tuning the Entire Model

In this approach, you replace the final layer as before, but instead of freezing the rest of the model, you continue to train (or "fine-tune") all the layers.

Typically, you would use a much **smaller learning rate** for the pre-trained layers than for the new classification layer. This is because the pre-trained layers already contain useful features, and you only want to adjust them slightly to better fit your new data, rather than changing them drastically.

**Steps:**
1.  Load the pre-trained model.
2.  Replace the final layer.
3.  Set up an optimizer that has different learning rates for different parameter groups (one group for the pre-trained layers, one for the new layer).
4.  Train the entire model.

This approach is more powerful if you have a larger dataset, but it also requires more computational resources.

The `transfer_learning_example.py` script provides a complete example of using a pre-trained ResNet model for feature extraction on a custom dataset.
