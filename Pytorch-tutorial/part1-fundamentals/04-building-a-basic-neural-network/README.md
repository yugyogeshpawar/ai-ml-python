# Part 1, Topic 4: Building a Basic Neural Network

You've learned about Tensors and Autograd. Now it's time to combine these concepts and build your first neural network in PyTorch using the `torch.nn` module.

## What is `torch.nn`?

`torch.nn` is PyTorch's module specifically designed for building neural networks. It provides a set of powerful tools, including:
-   **Layers:** Pre-defined layers like linear (fully connected), convolutional, and recurrent layers.
-   **Activation Functions:** Non-linear functions like ReLU, Sigmoid, and Tanh.
-   **Loss Functions:** Standard loss functions like Mean Squared Error and Cross-Entropy Loss.
-   **Containers:** A way to organize layers into a model, most importantly `nn.Module`.

## The `nn.Module` Class

In PyTorch, every neural network you build should be a class that inherits from `nn.Module`. This base class provides essential functionality, such as tracking all the model's layers and parameters.

A typical `nn.Module` has two main parts:
1.  **The `__init__` method:** This is where you define and initialize all the layers your network will use (e.g., `nn.Linear`, `nn.ReLU`).
2.  **The `forward` method:** This is where you define the "forward pass" of your network. You take an input tensor and pass it through the layers you defined in `__init__` to produce an output tensor.

`autograd` automatically builds the computation graph based on the operations in your `forward` method, so you don't need to define a `backward` method.

## Building a Simple Network

Let's build a simple neural network to see how this works. Our network will have:
-   An input layer.
-   One hidden layer with a ReLU activation function.
-   An output layer.

This is a standard feedforward neural network.

### Step 1: Define the Network Class

We create a class `SimpleNet` that inherits from `nn.Module`.

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Call the constructor of the parent class (nn.Module)
        super(SimpleNet, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size) # First fully connected layer
        self.relu = nn.ReLU() # ReLU activation function
        self.layer2 = nn.Linear(hidden_size, output_size) # Second fully connected layer

    def forward(self, x):
        # Define the forward pass
        # The order of operations here defines the network's architecture
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out
```

### Step 2: Create an Instance of the Network

Now, we can create an instance of our network by specifying the input, hidden, and output sizes.

```python
# Define network parameters
input_size = 10
hidden_size = 32
output_size = 1

# Create the model
model = SimpleNet(input_size, hidden_size, output_size)
print(model)
```

Printing the model will show you a summary of the layers it contains.

### Step 3: The Forward Pass

To use the model, you pass it an input tensor. The `forward` method is called automatically.

```python
# Create a random input tensor
# The input should have a shape of (batch_size, input_size)
input_tensor = torch.randn(64, input_size) # A batch of 64 samples

# Get the model's output
output = model(input_tensor)

print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

## Understanding Layers

### `nn.Linear(in_features, out_features)`

This is a fully connected layer. It applies a linear transformation to the incoming data: `y = xA^T + b`.
-   `in_features`: The size of each input sample.
-   `out_features`: The size of each output sample.

PyTorch automatically creates the weight (`A`) and bias (`b`) tensors for this layer, and they are automatically registered as model parameters.

### `nn.ReLU()`

This is a non-linear activation function. It applies the rectified linear unit function element-wise: `ReLU(x) = max(0, x)`. Activation functions are crucial for allowing neural networks to learn complex, non-linear patterns.

## Viewing Model Parameters

You can easily inspect the parameters of your model (the weights and biases that will be updated during training).

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")
```

These are the tensors that `autograd` will track and that an optimizer will update during the training process.

The `basic_neural_network_example.py` script provides a complete, runnable example of defining and using a simple neural network.
