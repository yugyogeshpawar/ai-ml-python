# Exercises: Building a Basic Neural Network

These exercises will help you practice defining and manipulating neural network models using `torch.nn`.

## Exercise 1: Build a Deeper Network

**Task:** Modify the `SimpleNet` class to create a "deeper" network. The new network should have **two** hidden layers instead of one.

1.  Create a new class called `DeeperNet`.
2.  The network architecture should be:
    -   Input layer
    -   First hidden layer (`nn.Linear`) with 128 neurons, followed by a `ReLU` activation.
    -   Second hidden layer (`nn.Linear`) with 64 neurons, followed by a `ReLU` activation.
    -   Output layer (`nn.Linear`).
3.  Instantiate the network and print its architecture. The `input_size` and `output_size` can be the same as in the example.

**Goal:** Practice adding more layers to a network. This demonstrates how to stack layers to increase model complexity.

## Exercise 2: Use a Different Activation Function

**Task:** Create a new network that uses the `Sigmoid` activation function instead of `ReLU`.

1.  Create a class `SigmoidNet`.
2.  Use the same architecture as the original `SimpleNet` (one hidden layer), but replace `nn.ReLU()` with `nn.Sigmoid()`.
3.  Instantiate the network, print its architecture, and perform a forward pass with a random tensor.

**Goal:** Learn how to swap out components like activation functions. This is a common task when experimenting with different model architectures.

## Exercise 3: Manually Inspect Layer Weights

**Task:** Write a script to access and print the shape of the weight and bias of the **first** linear layer (`layer1`) of the `SimpleNet` model.

1.  Create an instance of `SimpleNet`.
2.  Access the first linear layer directly (e.g., `model.layers[0]` if you used `nn.Sequential`, or `model.layer1` if you defined it as an attribute).
3.  The weights and biases are stored in the `.weight` and `.bias` attributes of the linear layer object.
4.  Print the shape of the weight tensor and the bias tensor.

**Goal:** Understand how to "look inside" a model and inspect the parameters of individual layers. This is useful for debugging, initialization, and analysis.
