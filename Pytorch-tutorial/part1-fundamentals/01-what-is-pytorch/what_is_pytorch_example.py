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
