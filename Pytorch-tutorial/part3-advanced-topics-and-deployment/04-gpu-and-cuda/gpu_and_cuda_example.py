# gpu_and_cuda_example.py

import torch
import torch.nn as nn
import time

def main():
    """
    Demonstrates the standard workflow for using a GPU and compares
    the computation time against a CPU.
    """
    
    # --- 1. Define the Device ---
    # Check if a CUDA-enabled GPU is available, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    if not torch.cuda.is_available():
        print("CUDA not available. This demo will run on the CPU.")
        print("Performance comparison will not be meaningful.")

    # --- 2. Create a Model and Tensors ---
    # A simple linear model
    model = nn.Linear(5000, 5000)
    
    # A large input tensor to make the computation substantial
    input_tensor = torch.randn(1000, 5000)

    # --- 3. CPU Computation ---
    print("\n--- Running on CPU ---")
    
    # The model and tensor are on the CPU by default
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input tensor device: {input_tensor.device}")
    
    start_time = time.time()
    # Perform a forward pass on the CPU
    output_cpu = model(input_tensor)
    end_time = time.time()
    
    cpu_time = end_time - start_time
    print(f"CPU computation finished in: {cpu_time:.4f} seconds")
    print(f"Output tensor is on device: {output_cpu.device}")

    # --- 4. GPU Computation ---
    if torch.cuda.is_available():
        print("\n--- Running on GPU ---")
        
        # a. Move the model to the GPU
        model.to(device)
        print(f"Model moved to device: {next(model.parameters()).device}")
        
        # b. Move the input tensor to the GPU
        input_tensor_gpu = input_tensor.to(device)
        print(f"Input tensor moved to device: {input_tensor_gpu.device}")
        
        # It's good practice to synchronize before timing GPU operations
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Perform a forward pass on the GPU
        output_gpu = model(input_tensor_gpu)
        
        # Synchronize again to ensure the operation is complete before stopping the timer
        torch.cuda.synchronize()
        end_time = time.time()
        
        gpu_time = end_time - start_time
        print(f"GPU computation finished in: {gpu_time:.4f} seconds")
        print(f"Output tensor is on device: {output_gpu.device}")

        # --- 5. Performance Comparison ---
        print("\n--- Performance Comparison ---")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        if gpu_time > 0:
            print(f"GPU is approximately {cpu_time / gpu_time:.2f}x faster for this operation.")

        # --- 6. Moving Data Back to CPU ---
        print("\n--- Moving Data Back to CPU ---")
        # To perform CPU-specific operations like converting to NumPy,
        # the tensor must be moved back to the CPU.
        
        # This would cause an error:
        # numpy_array = output_gpu.numpy() 
        
        # Correct way:
        output_back_on_cpu = output_gpu.cpu()
        print(f"Tensor moved back to device: {output_back_on_cpu.device}")
        
        # Now we can convert to NumPy
        numpy_array = output_back_on_cpu.numpy()
        print(f"Successfully converted tensor to NumPy array of shape: {numpy_array.shape}")

if __name__ == '__main__':
    main()
