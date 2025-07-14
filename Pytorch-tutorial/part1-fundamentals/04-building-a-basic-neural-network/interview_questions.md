# Interview Questions: Building a Basic Neural Network

These questions focus on the structure and components of a PyTorch model.

### 1. What is the role of `nn.Module` in PyTorch? Why do our custom network classes inherit from it?

**Answer:**
`nn.Module` is the base class for all neural network modules in PyTorch. Its role is to provide a framework for creating and managing neural networks.

We inherit from `nn.Module` for several critical reasons:
1.  **Parameter Tracking:** `nn.Module` automatically registers any `nn.Module` subclass (like `nn.Linear`, `nn.Conv2d`, etc.) assigned as an attribute in the `__init__` method. This allows it to track all the learnable parameters (weights and biases) of the network. You can easily access all parameters using the `.parameters()` or `.named_parameters()` methods.
2.  **State Management:** It helps manage the state of the model, such as switching between training and evaluation modes (`model.train()` and `model.eval()`). This is important for layers like Dropout and BatchNorm, which behave differently during training and inference.
3.  **Functionality:** It provides essential methods for managing the model, such as saving and loading the model's state (`.state_dict()`, `.load_state_dict()`), and moving the entire model to a different device (`.to(device)`).

By inheriting from `nn.Module`, we get all this powerful functionality for free, allowing us to focus on defining the architecture of our network in the `forward` method.

### 2. What is the difference between the `__init__` method and the `forward` method in an `nn.Module` class?

**Answer:**
-   The `__init__` method is the **constructor** of the class. Its purpose is to **define and initialize** all the components (layers, activation functions, etc.) that the network will use. This is where you create instances of `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, etc. These layers are created once when the model is first instantiated.

-   The `forward` method defines the **computation** of the network. Its purpose is to specify how the input data `x` flows through the layers that were defined in `__init__`. The `forward` method is executed every time you call the model on an input (e.g., `output = model(input_tensor)`). The sequence of operations in this method defines the actual architecture of the computation graph. `autograd` then builds its graph based on these operations to enable backpropagation.

In short: `__init__` sets up the building blocks, and `forward` assembles them to perform the calculation.

### 3. What is an activation function, and why is it necessary in a neural network? Give an example of one.

**Answer:**
An **activation function** is a function applied to the output of a neuron or a layer of neurons. Its primary purpose is to introduce **non-linearity** into the network.

**Why is it necessary?**
Without a non-linear activation function, a neural network, no matter how many layers it has, would just be a series of linear operations (matrix multiplications and additions). A composition of linear functions is itself a linear function. Therefore, without non-linearity, the network could only learn linear relationships between the input and output, which would be no more powerful than a simple linear regression model.

Non-linear activation functions allow the network to learn much more complex, non-linear patterns in the data, which is essential for tasks like image recognition, natural language processing, and more.

**Example:**
A common activation function is the **Rectified Linear Unit (ReLU)**, implemented in PyTorch as `nn.ReLU()`. It is defined as `f(x) = max(0, x)`. It replaces all negative values in the input tensor with zero and keeps positive values unchanged. It is computationally efficient and helps mitigate the vanishing gradient problem, making it a popular default choice for hidden layers.
