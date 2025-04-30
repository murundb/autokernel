import os
import time
import numpy as np
import json
from typing import Dict, List, Any, Optional, Callable
import cupy as cp
import torch

class CudaRunner:
    """
    Class to load, execute and time CUDA kernels using PyTorch.
    Requires PyTorch with CUDA support to be installed.
    This runner is kernel-agnostic and can run any CUDA kernel.
    """
    def __init__(self):
        try:
            if not torch.cuda.is_available():
                raise ImportError("CUDA is not available in PyTorch")
            
            self.torch = torch
            
            # Print device info
            device_id = 0
            print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(device_id)}")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("PyTorch with CUDA support is required.")
    
    def run_kernel(self, 
                  kernel_code: str,
                  kernel_name: str, 
                  input_setup_fn: Callable, 
                  verification_fn: Callable = None,
                  num_iterations: int = 5, 
                  block_size: tuple = (32, 32, 1),
                  grid_size: tuple = None) -> Dict[str, Any]:
        """
        Run the kernel and measure execution time
        
        Args:
            kernel_code: String containing the CUDA kernel code
            kernel_name: Name of the kernel function to run
            input_setup_fn: Function that returns a tuple of (args, grid_size)
                           where args is a list of arguments to pass to the kernel
                           and grid_size is the grid dimensions to use
            verification_fn: Function that takes the kernel outputs and returns True if correct
            num_iterations: Number of times to run the kernel for timing
            block_size: CUDA block dimensions (threads per block)
            grid_size: CUDA grid dimensions (optional, can be provided by input_setup_fn)
            
        Returns:
            Dict containing timing results
        """
        if not kernel_code:
            return {"error": "No kernel code provided"}
        
        try:
            # Create PyTorch CUDA kernel
            
            # Compile the kernel with CuPy
            module = cp.RawModule(
                code=kernel_code,
                name_expressions=[kernel_name]
            )
            kernel = module.get_function(kernel_name)
            
            # Get kernel inputs and grid size from the setup function
            args, dynamic_grid_size = input_setup_fn()
            
            # Use provided grid_size if given, otherwise use the one from input_setup_fn
            if grid_size is None:
                grid_size = dynamic_grid_size
            
            # Run multiple iterations and measure time
            execution_times = []
            
            for i in range(num_iterations):
                # Synchronize before timing
                self.torch.cuda.synchronize()
                start_time = time.time()
                
                # Launch kernel
                kernel(grid_size, block_size, args)
                
                # Synchronize and measure time
                self.torch.cuda.synchronize()
                end_time = time.time()
                
                elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
                execution_times.append(elapsed_time)
                
                print(f"Iteration {i+1}: {elapsed_time:.2f} ms")
            
            # Calculate statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            # Verify result if verification function is provided
            is_correct = verification_fn() if verification_fn else None
            
            timing_results = {
                "iterations": num_iterations,
                "average_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "all_times_ms": execution_times,
                "correct_result": is_correct,
                "block_size": block_size,
                "grid_size": grid_size
            }
            
            return timing_results
            
        except Exception as e:
            import traceback
            print(e)
            print(traceback.format_exc())
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    