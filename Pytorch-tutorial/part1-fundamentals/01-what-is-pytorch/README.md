# Part 1, Topic 1: What is PyTorch?

Welcome to the first lesson in our PyTorch tutorial! This section will introduce you to PyTorch, explain its core features and advantages, and guide you through setting up a working environment.

## What is PyTorch?

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR). It is one of the most popular deep learning frameworks, known for its simplicity, flexibility, and strong community support.

At its core, PyTorch provides two fundamental features:
1.  **Tensor Computing:** Similar to NumPy, PyTorch provides powerful multi-dimensional arrays called **tensors**. However, PyTorch tensors can be moved to a GPU to accelerate computing, which is crucial for training large deep learning models.
2.  **Automatic Differentiation:** PyTorch has a built-in automatic differentiation engine called `autograd`. It automatically computes gradients (or derivatives) for all operations performed on tensors, which is the backbone of how neural networks learn.

## Why Use PyTorch?

PyTorch has gained immense popularity for several key reasons:

-   **Simplicity and Pythonic Nature:** PyTorch's API is designed to feel intuitive and natural for Python developers. It integrates seamlessly with the Python data science ecosystem (e.g., NumPy, SciPy, and Cython).
-   **Dynamic Computation Graph:** Unlike some other frameworks that use static graphs, PyTorch uses a **dynamic computation graph**. This means the graph is built on-the-fly as operations are executed. This makes debugging easier and allows for more flexible model architectures, especially in fields like Natural Language Processing (NLP) where input lengths can vary.
-   **Strong GPU Acceleration:** Training deep learning models can be computationally expensive. PyTorch makes it easy to move your data and models to a GPU, drastically speeding up the training process.
-   **Rich Ecosystem and Community:** PyTorch has a vast ecosystem of tools and libraries, such as `torchvision` for computer vision, `torchtext` for NLP, and `torchaudio` for audio processing. The community is active and provides excellent support and resources.
-   **Easy Deployment:** With tools like **TorchServe** and **TorchScript**, deploying PyTorch models to production environments has become increasingly straightforward.

## Setting Up Your Environment

To get started, you need to install Python and PyTorch. We recommend using a virtual environment to keep your project dependencies isolated.

### Step 1: Install Python

If you don't have Python installed, download it from the [official Python website](https://www.python.org/downloads/). PyTorch is compatible with Python 3.8 and later.

### Step 2: Create a Virtual Environment

A virtual environment prevents conflicts between different projects' dependencies. Open your terminal and run the following commands:

```bash
# Create a directory for your project
mkdir pytorch-tutorial
cd pytorch-tutorial

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install PyTorch

The best way to install PyTorch is by using the official command generator on the [PyTorch website](https://pytorch.org/get-started/locally/). This ensures you get the correct version for your operating system and hardware (CPU or GPU with CUDA).

For a typical setup on macOS or Windows (CPU-only), the command will look like this:

```bash
pip3 install torch torchvision torchaudio
```

If you have an NVIDIA GPU and want to enable CUDA acceleration, the command will be different. Please visit the official website to get the precise command for your system.

## Step-by-Step Code Tutorial

Let's walk through the example script `what_is_pytorch_example.py` to verify your installation.

### `what_is_pytorch_example.py`

This script is a simple "Hello, PyTorch!" check. It imports the `torch` library, prints the installed version, and creates a basic tensor.

```python
# what_is_pytorch_example.py

import torch

def main():
    """
    A simple script to verify the PyTorch installation.
    """
    # 1. Print PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # 2. Check if CUDA (GPU support) is available
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")

    if is_cuda_available:
        # Print the name of the current GPU
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # 3. Create a simple tensor
    # A tensor is a multi-dimensional array, the fundamental data structure in PyTorch.
    x = torch.tensor([1, 2, 3])
    print(f"\nCreated a simple tensor: {x}")
    print(f"Tensor shape: {x.shape}")
    print(f"Tensor data type: {x.dtype}")

if __name__ == "__main__":
    main()
```

### How to Run the Script

1.  Make sure your virtual environment is activated.
2.  Save the code above as `what_is_pytorch_example.py`.
3.  Run the script from your terminal:

    ```bash
    python what_is_pytorch_example.py
    ```

### Expected Output

You should see an output similar to this (the version number may vary):

```
PyTorch Version: 2.1.0
CUDA Available: False

Created a simple tensor: tensor([1, 2, 3])
Tensor shape: torch.Size([3])
Tensor data type: torch.int64
```

If you see this output, congratulations! Your PyTorch environment is set up correctly, and you are ready to move on to the next lesson.
