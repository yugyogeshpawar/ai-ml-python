# Exercises: What is PyTorch?

These exercises are designed to test your understanding of the setup process and basic environment verification.

## Exercise 1: Environment Setup and Verification

**Task:** Follow the steps in the `README.md` to set up your Python virtual environment and install PyTorch.

1.  Create a new directory for this tutorial.
2.  Inside that directory, create and activate a Python virtual environment.
3.  Install `torch`, `torchvision`, and `torchaudio` using `pip`.
4.  Run the `what_is_pytorch_example.py` script.

**Goal:** Successfully run the script and see the PyTorch version and tensor information printed to your console. This confirms your environment is correctly configured.

## Exercise 2: Modify the Example Script

**Task:** Modify the `what_is_pytorch_example.py` script to create and print a different tensor.

1.  Open the `what_is_pytorch_example.py` file.
2.  Change the tensor `x` from a 1D tensor `[1, 2, 3]` to a 2D tensor (a matrix). For example, you can create a 2x3 tensor (2 rows, 3 columns).
    ```python
    # Example of a 2x3 tensor
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    ```
3.  Print the new tensor, its shape, and its data type.

**Goal:** Understand how to create tensors with different shapes and verify their properties. The output should reflect the new tensor's dimensions.

## Exercise 3: Explore Your PyTorch Installation

**Task:** Add code to the script to find out more about your PyTorch installation.

1.  PyTorch has a function `torch.get_num_threads()` that returns the number of threads PyTorch is currently using for parallel processing on the CPU.
2.  Add a print statement to the `main()` function to display this information.

**Goal:** Learn how to explore the configuration and capabilities of your PyTorch installation programmatically.
