# Interview Questions: What is PyTorch?

These questions cover fundamental concepts about PyTorch's role, its core features, and how it compares to other frameworks.

### 1. What is PyTorch, and what are its two main features?

**Answer:**
PyTorch is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing. It was developed by Facebook's AI Research lab (FAIR).

Its two main features are:
1.  **Tensor Computing:** It provides a powerful N-dimensional array object called a `Tensor`, which is similar to NumPy's `ndarray` but can be used on a GPU to accelerate computing.
2.  **Automatic Differentiation:** PyTorch has a built-in module called `autograd` that builds a dynamic computation graph and automatically calculates gradients for any operation performed on tensors. This is essential for training neural networks via backpropagation.

### 2. What is a "dynamic computation graph," and why is it an advantage?

**Answer:**
A **dynamic computation graph** means that the graph representing the model's computations is built on-the-fly as the code is executed. Each time a model processes an input, a new graph is created.

This is a significant advantage for several reasons:
-   **Flexibility:** It is particularly useful for models where the architecture can change based on the input data, such as Recurrent Neural Networks (RNNs) in NLP, where sequence lengths can vary.
-   **Easier Debugging:** Because the graph is defined by the forward pass execution, you can use standard Python debugging tools like `pdb` or print statements to inspect values and troubleshoot issues at any point in the model. This is much more difficult with static graphs, where the graph is compiled first and then executed.
-   **Readability:** The code often looks more like standard Python, making it more intuitive and easier to understand for developers.

### 3. How does PyTorch's `Tensor` object differ from a NumPy `ndarray`?

**Answer:**
While PyTorch's `Tensor` and NumPy's `ndarray` are both powerful multi-dimensional array structures, they have key differences that make `Tensor` more suitable for deep learning:

1.  **GPU Acceleration:** The most significant difference is that PyTorch Tensors can be moved to a GPU (`.to('cuda')`) to perform massive parallel computations, which is critical for accelerating the training of deep learning models. NumPy arrays are limited to CPU computations.
2.  **Automatic Differentiation:** PyTorch Tensors are integrated with `autograd`. If a tensor has `requires_grad=True`, PyTorch will automatically track all operations on it to compute gradients during backpropagation. NumPy arrays do not have this capability.
3.  **Deep Learning Ecosystem:** Tensors are the fundamental data structure used throughout the PyTorch ecosystem, including its neural network layers (`torch.nn`), optimizers (`torch.optim`), and data loaders (`torch.utils.data`).

Despite these differences, PyTorch and NumPy are highly interoperable. You can convert a PyTorch Tensor to a NumPy array (`.numpy()`) and vice-versa (`torch.from_numpy()`) efficiently, allowing developers to leverage both libraries in their workflows.
