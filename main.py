import anthropic
import os
from dotenv import load_dotenv
from src.runners.cuda_runner import CudaRunner
import numpy as np
import torch
import cupy as cp
        
from src.gpu_types import GPUType

load_dotenv()  # take environment variables

# Load your API key securely
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def generate_kernel(gpu_type, task_description, constraints=None, manual_context=None):

    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software

    system_prompt = (
        "You are an expert {0} {1} GPU engineer. "
        "Your job is correctness and holding to the given task specification. "
        "Your main task is to generate highly efficient {2} kernels for {3} {4} GPUs. "
        "Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. Focus on minimizing execution speed, minimizing global memory access, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. "
        "Use precise, hardware-conscious code generation, following {5}'s architectural best practices.".format(gpu_manufacturer, gpu_hardware, gpu_software, gpu_manufacturer, gpu_hardware, gpu_hardware)
    )

    base_prompt = f"""
    Refer to the following architecture context (extracted from the manual):

    {manual_context if manual_context else "No additional context provided."}

    ---

    Target Task:
    {task_description}

    Constraints:
    {constraints if constraints else "No specific constraints provided."}

    ---

    Instructions:
    - Propose a new {gpu_software} kernel which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result. The name of the kernel should be matrixMultiply.
    - Return only the raw code, no explanations, no markdown formatting, no extra commentary given this will copied directly to the kernel file. Do not include language markers, code block formatting, or triple backticks in your response. Return only the raw code.
    """

    response = client.messages.create(
        model="claude-3-5-haiku-latest",   # You can change model if you want (haiku, sonnet, opus)
        max_tokens=2000,
        temperature=1,
        system=system_prompt,
        messages=[
            {"role": "user", "content": base_prompt}
        ]
    )

    generated_kernel_code = response.content[0].text
    # generated_kernel_code = '__global__ void matrixMultiply(float* A, float* B, float* C, int N) {\n    __shared__ float sharedA[32][32];\n    __shared__ float sharedB[32][32];\n    \n    int bx = blockIdx.x; \n    int by = blockIdx.y;\n    int tx = threadIdx.x;\n    int ty = threadIdx.y;\n    \n    int row = by * 32 + ty;\n    int col = bx * 32 + tx;\n    \n    float sum = 0.0f;\n    \n    for (int m = 0; m < (N + 31) / 32; ++m) {\n        if (row < N && m * 32 + tx < N)\n            sharedA[ty][tx] = A[row * N + m * 32 + tx];\n        else\n            sharedA[ty][tx] = 0.0f;\n        \n        if (col < N && m * 32 + ty < N)\n            sharedB[ty][tx] = B[(m * 32 + ty) * N + col];\n        else\n            sharedB[ty][tx] = 0.0f;\n        \n        __syncthreads();\n        \n        for (int k = 0; k < 32; ++k)\n            sum += sharedA[ty][k] * sharedB[k][tx];\n        \n        __syncthreads();\n    }\n    \n    if (row < N && col < N)\n        C[row * N + col] = sum;\n}'

    return generated_kernel_code

def run_and_time_kernel(kernel_code, matrix_dim=4096):
    try:
        runner = CudaRunner()
        
        # Setup function for matrix multiplication
        def matmul_setup():
            # Create input matrices
            a_host = torch.ones((matrix_dim, matrix_dim), dtype=torch.float32)
            b_host = torch.ones((matrix_dim, matrix_dim), dtype=torch.float32)
            
            # Move to GPU
            a_gpu = a_host.cuda()
            b_gpu = b_host.cuda()
            c_gpu = torch.zeros((matrix_dim, matrix_dim), dtype=torch.float32, device="cuda")
            
            # Calculate grid dimensions
            block_size = (32, 32, 1)  # Default block size
            grid_x = (matrix_dim + block_size[0] - 1) // block_size[0]
            grid_y = (matrix_dim + block_size[1] - 1) // block_size[1]
            grid = (grid_x, grid_y)
            
            # Store these for verification
            runner.a_gpu = a_gpu
            runner.b_gpu = b_gpu
            runner.c_gpu = c_gpu
            runner.matrix_dim = matrix_dim
            
            # Return kernel arguments and grid size
            return [int(a_gpu.data_ptr()), 
                    int(b_gpu.data_ptr()), 
                    int(c_gpu.data_ptr()), 
                    np.int32(matrix_dim)], grid
        
        # Verification function for matrix multiplication
        def verify_matmul():
            # Copy result back to CPU
            c_host = runner.c_gpu.cpu()
            
            # For matrices filled with 1.0, each element in C should be matrix_dim
            expected_value = float(matrix_dim)
            
            # Check a few elements
            sample_indices = [
                (0, 0), (0, 1), (1, 0), 
                (matrix_dim//2, matrix_dim//2),
                (matrix_dim-1, matrix_dim-1)
            ]
            
            all_correct = True
            for i, j in sample_indices:
                if abs(c_host[i, j].item() - expected_value) > 1e-5:
                    print(f"Verification failed at [{i}, {j}]: Expected {expected_value}, got {c_host[i, j].item()}")
                    all_correct = False
                    break
            
            if all_correct:
                print("Result verification passed for sampled elements")
            
            return all_correct
        
        # Run the kernel with our setup and verification functions
        return runner.run_kernel(
            kernel_code=kernel_code,
            kernel_name="matrixMultiply",
            input_setup_fn=matmul_setup,
            verification_fn=verify_matmul,
            num_iterations=5
        )
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":

    gpu_type = GPUType.Nvidia
    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software

    task = "Write {0} kernel that performs 4096x4096 matrix multiplication optimized for {1} {2} architecture.".format(gpu_software, gpu_manufacturer, gpu_hardware)
    constraints = (
        "Minimize global memory reads and writes. "
        + "Maximize usage of {} compute units without exceeding register limits. ".format(gpu_hardware)
        + "Minimize the processing time on the GPU. "
        + "Maximize GPU utilization."
    )

    manual_context = None

    kernel_code = generate_kernel(gpu_type, task, constraints, manual_context)
    # Run and time the kernel directly without saving to a file
    timing_data = run_and_time_kernel(kernel_code)
    # Save the generated kernel to a file
    output_filename = "generated_kernel.metal"
    with open(output_filename, "w") as f:
        f.write(kernel_code)