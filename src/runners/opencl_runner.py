import time
import numpy as np
import pyopencl as cl

class OpenclRunner:
    """
    Class to load, execute and time OpenCL kernels.
    Requires PyOpenCL to be installed.
    This runner is kernel-agnostic and can run any OpenCL kernel.
    """
    def __init__(self, platform_idx=0, device_idx=0):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found.")
        self.platform = platforms[platform_idx]
        self.device = self.platform.get_devices()[device_idx]
        print(f"Using OpenCL device: {self.device.name}")
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

    def run_kernel(self, 
                  kernel_code: str,
                  kernel_name: str, 
                  input_setup_fn, 
                  verification_fn=None,
                  num_iterations: int = 5, 
                  block_size: tuple = (32, 32, 1),
                  grid_size: tuple = None) -> dict:
        """
        Run the kernel and measure execution time.
        Args are similar to CudaRunner.
        """
        if not kernel_code:
            return {"error": "No kernel code provided"}

        try:
            # Build the kernel
            program = cl.Program(self.ctx, kernel_code).build()
            kernel = getattr(program, kernel_name)

            # Get kernel inputs and grid size from the setup function
            args, dynamic_grid_size = input_setup_fn(self.ctx)
            if grid_size is None:
                grid_size = dynamic_grid_size
            
            if isinstance(grid_size, int):
                grid_size = [grid_size]
            if isinstance(block_size, int):
                block_size = [block_size]

            execution_times = []

            for i in range(num_iterations):
                self.queue.finish()
                start_time = time.time()
                # Launch kernel
                global_size = grid_size
                local_size = block_size
                kernel(self.queue, global_size, local_size, *args)
                self.queue.finish()
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                execution_times.append(elapsed_time)
                print(f"Iteration {i+1}: {elapsed_time:.2f} ms")

            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)

            is_correct, msg = verification_fn() if verification_fn else (None, None)

            timing_results = {
                "iterations": num_iterations,
                "average_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "all_times_ms": execution_times,
                "correct_result": is_correct,
                "verification_feedback": msg,
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