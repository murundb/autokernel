import anthropic
import os
from dotenv import load_dotenv
from src.runners.cuda_runner import CudaRunner
from src.runners.opencl_runner import OpenclRunner
import numpy as np
import torch
import cupy as cp
import pyopencl as cl
        
from src.gpu_types import GPUType
import json

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
        "Structure your output in JSON format strictly."
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
    - Return a python dictionary in JSON format with key "kernel_code" which contains the raw code for the kernel, the global size and local size in the "global_size" and "local_size" keys. I want no explanations, no markdown formatting, no extra commentary given this will be copied directly to the kernel file. Do not include language markers, code block formatting, or triple backticks in your response. Return only the raw code, global size, and local size.
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

    generated_kernel_config = response.content[0].text
    # generated_kernel_config = '__global__ void matrixMultiply(float* A, float* B, float* C, int N) {\n    __shared__ float sharedA[32][32];\n    __shared__ float sharedB[32][32];\n    \n    int bx = blockIdx.x; \n    int by = blockIdx.y;\n    int tx = threadIdx.x;\n    int ty = threadIdx.y;\n    \n    int row = by * 32 + ty;\n    int col = bx * 32 + tx;\n    \n    float sum = 0.0f;\n    \n    for (int m = 0; m < (N + 31) / 32; ++m) {\n        if (row < N && m * 32 + tx < N)\n            sharedA[ty][tx] = A[row * N + m * 32 + tx];\n        else\n            sharedA[ty][tx] = 0.0f;\n        \n        if (col < N && m * 32 + ty < N)\n            sharedB[ty][tx] = B[(m * 32 + ty) * N + col];\n        else\n            sharedB[ty][tx] = 0.0f;\n        \n        __syncthreads();\n        \n        for (int k = 0; k < 32; ++k)\n            sum += sharedA[ty][k] * sharedB[k][tx];\n        \n        __syncthreads();\n    }\n    \n    if (row < N && col < N)\n        C[row * N + col] = sum;\n}'
    parsed_config = json.loads(generated_kernel_config)
    
    return parsed_config

def run_and_time_kernel(kernel_config, matrix_dim=4096, backend="cuda"):
    try:
        if backend == "cuda":
            runner = CudaRunner()
            def matmul_setup():
                a_host = torch.ones((matrix_dim, matrix_dim), dtype=torch.float32)
                b_host = torch.ones((matrix_dim, matrix_dim), dtype=torch.float32)
                a_gpu = a_host.cuda()
                b_gpu = b_host.cuda()
                c_gpu = torch.zeros((matrix_dim, matrix_dim), dtype=torch.float32, device="cuda")
                block_size = (32, 32, 1)
                grid_x = (matrix_dim + block_size[0] - 1) // block_size[0]
                grid_y = (matrix_dim + block_size[1] - 1) // block_size[1]
                grid = (grid_x, grid_y)
                runner.a_gpu = a_gpu
                runner.b_gpu = b_gpu
                runner.c_gpu = c_gpu
                runner.matrix_dim = matrix_dim
                return [int(a_gpu.data_ptr()), int(b_gpu.data_ptr()), int(c_gpu.data_ptr()), np.int32(matrix_dim)], grid

            def verify_matmul():
                c_host = runner.c_gpu.cpu()
                expected_value = float(matrix_dim)
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

            return runner.run_kernel(
                kernel_code=kernel_config["kernel_code"],
                kernel_name="matrixMultiply",
                input_setup_fn=matmul_setup,
                verification_fn=verify_matmul,
                num_iterations=5
            )

        elif backend == "opencl":
            runner = OpenclRunner()
            def matmul_setup(ctx):
                mf = cl.mem_flags
                a_host = np.ones((matrix_dim, matrix_dim), dtype=np.float32)
                b_host = np.ones((matrix_dim, matrix_dim), dtype=np.float32)
                c_host = np.zeros((matrix_dim, matrix_dim), dtype=np.float32)
                a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
                b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)
                c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c_host.nbytes)
                runner.a_buf = a_buf
                runner.b_buf = b_buf
                runner.c_buf = c_buf
                runner.matrix_dim = matrix_dim
                runner.c_host = c_host
                return [a_buf, b_buf, c_buf, np.int32(matrix_dim)], None

            def verify_matmul():
                cl.enqueue_copy(runner.queue, runner.c_host, runner.c_buf)
                expected_value = float(matrix_dim)
                sample_indices = [
                    (0, 0), (0, 1), (1, 0),
                    (matrix_dim//2, matrix_dim//2),
                    (matrix_dim-1, matrix_dim-1)
                ]
                all_correct = True
                for i, j in sample_indices:
                    if abs(runner.c_host[i, j] - expected_value) > 1e-5:
                        print(f"Verification failed at [{i}, {j}]: Expected {expected_value}, got {runner.c_host[i, j]}")
                        all_correct = False
                        break
                if all_correct:
                    print("Result verification passed for sampled elements")
                return all_correct

            grid = kernel_config["global_size"]
            block_size = kernel_config["local_size"]
            
            return runner.run_kernel(
                kernel_code=kernel_config["kernel_code"],
                kernel_name="matrixMultiply",
                input_setup_fn=lambda ctx: matmul_setup(ctx),
                verification_fn=verify_matmul,
                num_iterations=5,
                block_size=block_size,
                grid_size=grid
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Choose backend: "cuda" or "opencl"
    # backend = os.environ.get("AUTOKERNEL_BACKEND", "cuda")
    backend = "opencl"

    def get_device_info(backend):
        if backend == "cuda":
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(0)
            return {
                "Device Name": device.name(),
                "Compute Capability": f"{device.compute_capability()}",
                "Total Memory": f"{device.total_memory() // (1024*1024)} MB",
                "Clock Rate": f"{device.clock_rate / 1000:.2f} MHz",  # <-- no ()
                "Max Threads Per Block": device.max_threads_per_block,
                "Max Block Dimensions": device.max_block_dim_x,
                "Max Grid Dimensions": device.max_grid_dim_x,
                "Warp Size": device.warp_size,
                "Max Shared Memory Per Block": f"{device.max_shared_memory_per_block // 1024} KB",
                "Max Registers Per Block": device.max_registers_per_block,
                "Async Engine Count": device.async_engine_count
            }
        else:  # OpenCL
            import pyopencl as cl
            platforms = cl.get_platforms()
            devices = platforms[0].get_devices()
            device = devices[0]
            return {
                "Device Name": device.name,
                "Device Type": cl.device_type.to_string(device.type),
                "Device Vendor": device.vendor,
                "Device Version": device.version,
                "Driver Version": device.driver_version,
                "Max Compute Units": device.max_compute_units,
                "Max Work Group Size": device.max_work_group_size,
                "Global Memory Size": f"{device.global_mem_size // (1024*1024)} MB",
                "Local Memory Size": f"{device.local_mem_size // 1024} KB",
                "Max Work Item Dimensions": device.max_work_item_dimensions,
                "Max Work Item Sizes": device.max_work_item_sizes,
                "Max Clock Frequency": f"{device.max_clock_frequency} MHz",
                "Image Support": device.image_support,
                "Extensions": device.extensions.split(),
                "Preferred Vector Width Float": device.preferred_vector_width_float,
                "Native Vector Width Float": device.native_vector_width_float
            }
    gpu_type = GPUType.Nvidia if backend == "cuda" else GPUType.Qualcomm  # Example: use AMD for OpenCL
    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software

    device_info = get_device_info(backend)
    print("\nDevice Information:")
    device_info_str = ""
    for key, value in device_info.items():
        device_info_str  += f"{key}: {value}\n"
    
    task = "Write {0} kernel that performs 4096x4096 matrix multiplication optimized for {1} {2} architecture.".format(gpu_software, gpu_manufacturer, gpu_hardware)
    constraints = (
        "Minimize global memory reads and writes. "
        + "Maximize usage of {} compute units without exceeding register limits. ".format(gpu_hardware)
        + "Minimize the processing time on the GPU. "
        + "Maximize GPU utilization. "
        + f"Device Information: {device_info_str}"
    )

    manual_context = None

    kernel_config = generate_kernel(gpu_type, task, constraints, manual_context)
    timing_data = run_and_time_kernel(kernel_config, backend=backend)
    output_filename = f"generated_kernel.{gpu_software.lower()}"
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(json.dumps(kernel_config, indent=4, ensure_ascii=False))