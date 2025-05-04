import json
import numpy as np
import torch
import pyopencl as cl

from src.runners.cuda_runner import CudaRunner
from src.runners.opencl_runner import OpenclRunner
from src.gpu_types import GPUType

class KernelManager:
    def __init__(self, llm, tokenizer, max_tokens=8000):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def generate_kernel(self, gpu_type, task_description, constraints=None, manual_context=None, history=None):
        gpu_manufacturer = gpu_type.name
        gpu_hardware = gpu_type.value.hardware
        gpu_software = gpu_type.value.software

        system_prompt = (
            f"You are an expert {gpu_manufacturer} {gpu_hardware} GPU engineer. "
            f"Your job is correctness and holding to the given task specification. "
            f"Your main task is to generate highly efficient {gpu_software} kernels for {gpu_manufacturer} {gpu_hardware} GPUs. "
            "Structure your output in JSON format strictly."
            "Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. "
            "Focus on minimizing execution speed, minimizing global memory access, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. "
            f"Use precise, hardware-conscious code generation, following {gpu_hardware}'s architectural best practices."
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
        - Propose a new {gpu_software} kernel which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result. 
        - Return a python dictionary in JSON format with key "kernel_code" which contains the raw code for the kernel, the global size and local size in the "global_size" and "local_size" keys. I want no explanations, no markdown formatting, no extra commentary given this will be copied directly to the kernel file. Do not include language markers, code block formatting, or triple backticks in your response. Return only the raw code, global size, and local size.
        """

        messages = history.history.copy() if history else []
        messages.append({"role": "user", "content": base_prompt})

        # Trim history to fit context window
        if history:
            history.trim_to_fit(self.max_tokens, self.tokenizer)

        response = self.llm.chat(messages, system_prompt=system_prompt)
        if history:
            history.add("assistant", response)
        return json.loads(response)

    def run_and_time_kernel(self, kernel_config, matrix_dim=4096, backend="cuda"):
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