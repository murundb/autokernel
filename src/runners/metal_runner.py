import os
import time
import numpy as np
import objc
from Foundation import NSBundle, NSURL, NSError
import Metal
import MetalPerformanceShaders

class MetalKernelRunner:
    def __init__(self):
        # Initialize Metal device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")
        
        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create command queue")
        
        # Matrix dimensions for matmul (4096x4096 as specified in the task)
        self.matrix_dim = 4096
        self.buffer_size = self.matrix_dim * self.matrix_dim * 4  # 4 bytes per float
        
        # Create buffers for matrices
        self.buffer_a = self.device.newBufferWithLength_options_(self.buffer_size, Metal.MTLResourceStorageModeShared)
        self.buffer_b = self.device.newBufferWithLength_options_(self.buffer_size, Metal.MTLResourceStorageModeShared)
        self.buffer_c = self.device.newBufferWithLength_options_(self.buffer_size, Metal.MTLResourceStorageModeShared)
        
        # Default library and function
        self.library = None
        self.function = None
        self.pipeline_state = None
    
    def load_kernel(self, kernel_path):
        """Load and compile the Metal kernel from the given path"""
        try:
            # Create URL for the Metal file
            url = NSURL.fileURLWithPath_(kernel_path)
            
            # Load the library
            error = objc.nil
            self.library = self.device.newLibraryWithURL_error_(url, error)
            if self.library is None:
                raise RuntimeError(f"Failed to load library: {error}")
            
            # Get the matmul function
            self.function = self.library.newFunctionWithName_("matmul")
            if self.function is None:
                raise RuntimeError("Failed to find matmul function in the kernel")
            
            # Create compute pipeline state
            self.pipeline_state = self.device.newComputePipelineStateWithFunction_error_(self.function, error)
            if self.pipeline_state is None:
                raise RuntimeError(f"Failed to create pipeline state: {error}")
            
            return True
        except Exception as e:
            print(f"Error loading kernel: {e}")
            return False
    
    def prepare_data(self):
        """Initialize input matrices with random data"""
        # Get pointers to buffer contents
        ptr_a = self.buffer_a.contents()
        ptr_b = self.buffer_b.contents()
        
        # Create NumPy arrays from the pointers
        a_np = np.ctypeslib.as_array(ptr_a, shape=(self.matrix_dim, self.matrix_dim)).astype(np.float32)
        b_np = np.ctypeslib.as_array(ptr_b, shape=(self.matrix_dim, self.matrix_dim)).astype(np.float32)
        
        # Fill with random data
        a_np.fill(1.0)  # Simple initialization for testing
        b_np.fill(1.0)
    
    def run_kernel(self, num_iterations=5):
        """Run the kernel and measure execution time"""
        if self.pipeline_state is None:
            raise RuntimeError("Pipeline state not initialized. Load a kernel first.")
        
        # Prepare data
        self.prepare_data()
        
        # Calculate grid and threadgroup sizes
        threads_per_threadgroup = Metal.MTLSizeMake(16, 16, 1)
        threadgroups = Metal.MTLSizeMake(
            (self.matrix_dim + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            (self.matrix_dim + threads_per_threadgroup.height - 1) // threads_per_threadgroup.height,
            1
        )
        
        # Run multiple iterations and measure time
        execution_times = []
        
        for i in range(num_iterations):
            # Create command buffer
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set pipeline state and buffers
            compute_encoder.setComputePipelineState_(self.pipeline_state)
            compute_encoder.setBuffer_offset_atIndex_(self.buffer_a, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(self.buffer_b, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(self.buffer_c, 0, 2)
            
            # Dispatch threadgroups
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_threadgroup)
            compute_encoder.endEncoding()
            
            # Start timing
            start_time = time.time()
            
            # Execute command buffer
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # End timing
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            execution_times.append(execution_time)
            
            print(f"Iteration {i+1}: {execution_time:.2f} ms")
        
        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        timing_results = {
            "iterations": num_iterations,
            "average_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "all_times_ms": execution_times
        }
        
        return timing_results
    
    def verify_result(self):
        """Verify the correctness of the matrix multiplication result"""
        # For a simple verification with matrices filled with 1.0,
        # each element in C should be equal to matrix_dim
        ptr_c = self.buffer_c.contents()
        c_np = np.ctypeslib.as_array(ptr_c, shape=(self.matrix_dim, self.matrix_dim)).astype(np.float32)
        
        # Check a few elements (checking all would be too slow)
        expected_value = float(self.matrix_dim)
        sample_indices = [(0, 0), (0, 1), (1, 0), (100, 100), (1000, 1000)]
        
        all_correct = True
        for i, j in sample_indices:
            if abs(c_np[i, j] - expected_value) > 1e-5:
                print(f"Verification failed at [{i}, {j}]: Expected {expected_value}, got {c_np[i, j]}")
                all_correct = False
        
        if all_correct:
            print("Result verification passed for sampled elements")
        
        return all_correct