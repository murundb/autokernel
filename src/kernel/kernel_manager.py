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
        self.cuda_runner = CudaRunner()
        self.opencl_runner = OpenclRunner()

    def generate_kernels(self, gpu_type, task_description, constraints=None, manual_context=None, history=None):
        gpu_manufacturer = gpu_type.name
        gpu_hardware = gpu_type.value.hardware
        gpu_software = gpu_type.value.software

        system_prompt = (
            f"You are an expert {gpu_manufacturer} {gpu_hardware} GPU engineer. "
            f"Your job is correctness and holding to the given task specification. "
            f"Your main task is to generate highly efficient {gpu_software} kernels for {gpu_manufacturer} {gpu_hardware} GPUs. "
            "Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. "
            "Focus on minimizing execution speed, minimizing global memory access, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. "
            f"Use precise, hardware-conscious code generation, following {gpu_hardware}'s architectural best practices."
            """Structure your output in JSON format strictly. Return output as a single-line JSON string where every newline is written as "\n" (two characters), not as an actual line break."""
        )

        base_prompt = f"""
        {"Refer to the following architecture context (extracted from the manual): "+manual_context if manual_context else ""}

        ---

        Target Task:
        {task_description}

        Constraints:
        {constraints if constraints else "No specific constraints provided."}
        ---

         Instructions:
        - Propose as many {gpu_software} kernels as needed to accomplish the task. The kernels should aim to reduce the runtime of the task, while ensuring the final output returns the correct result.""" + """
        - For each kernel, return:
            - "runner_setup": Sets up the runner for this kernel
        - For example for computing 2*A + B on two matrices A and B we can have one kernel that does the compute or we can split it into two kernels. Below is the example of two kernels accomplishing 2*A+B of size 4096*4096:
            "runner_setup":
                "
                def runner_setup():
                    import time
                    import numpy as np
                    import pyopencl as cl
    
                    # Add the kermnel code into a python string
                    opencl_code = "__kernel void scalarMultiply(const int scalar, __global const float* restrict A, __global float* restrict intermediate_result, const int matrix_dim) {// kernel code here} \n __kernel void matrixAdd(__global const float* restrict intermediate_result, __global const float* restrict B, __global float* restrict result, const int matrix_dim) {// kernel code here}"
    
                    dim = 1024

                    # Function to verify the results
                    def verification_fn():
                        result_host = np.empty_like(A_host)
                        cl.enqueue_copy(queue, result_host, C_buf)
                        expected = 2*A_host + B_host
                        ok = np.allclose(result_host, expected, rtol=1e-5, atol=1e-5)
                        return ok, "Results match!" if ok else "Results do NOT match!"

                    # Initialize data with random values
                    A_host = np.random.rand(dim, dim).astype(np.float32)
                    B_host = np.random.rand(dim, dim).astype(np.float32)
                    
                    # Create OpenCL buffers
                    mf = cl.mem_flags
                    platforms = cl.get_platforms()
                    if not platforms:
                        raise RuntimeError("No OpenCL platforms found.")
                    platform = platforms[0]
                    device = platform.get_devices()[0]
                    print(f"Using OpenCL device: {device.name}")
                    ctx = cl.Context([device])
                    queue = cl.CommandQueue(ctx)

                    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
                    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
                    intermediate_result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=dim * dim * np.dtype(np.float32).itemsize)
                    result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=dim * dim * np.dtype(np.float32).itemsize)
                    
                    # Build the kernel
                    program = cl.Program(ctx, opencl_code).build()
                    scalarMultiply = getattr(program, "scalarMultiply")
                    matrixAdd = getattr(program, "matrixAdd")

                    execution_times = []
                    for i in range(num_iterations):
                        # Get kernel inputs and grid size from the setup function
                        args = (2, A_buf, intermediate_result_buf, np.int32(dim))
                        # Normalise grid / block specification to sequences
                        global_size = [4096]
                        local_size = [512]
                        queue.finish()
                        start_time = time.time()
                        # Launch kernel
                        scalarMultiply(queue, global_size, local_size, *args)
                        queue.finish()
                        end_time = time.time()
                        elapsed_time = (end_time - start_time) * 1000

                        # Get kernel inputs and grid size from the setup function
                        args = (intermediate_result_buf, B_buf, result_buf, np.int32(dim))
                        # Normalise grid / block specification to sequences
                        global_size = [4096]
                        local_size = [1024]

                        queue.finish()
                        start_time = time.time()
                        # Launch kernel
                        matrixAdd(queue, global_size, local_size, *args)
                        queue.finish()
                        end_time = time.time()
                        elapsed_time += (end_time - start_time) * 1000

                        execution_times.append(elapsed_time)
                        print(f"Iteration {i+1}: {elapsed_time:.2f} ms")

                    avg_time = sum(execution_times) / len(execution_times)
                    min_time = min(execution_times)
                    max_time = max(execution_times)

                    is_correct, msg = verification_fn() if verification_fn else (None, None)

                    timing_result = {
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
                    return timing_result
                    ",
        - Return a python dictionary in JSON format with key "kernels", whose value is a list of dictionaries as described above.
        - Do not include explanations, markdown formatting, or extra commentary. Do not include language markers, code block formatting, triple backticks, or any other quotation or formatting around the JSON output. Return only the raw JSON.
        """

        messages = history.history.copy() if history else []
        messages.append({"role": "user", "content": base_prompt})

        # Trim history to fit context window
        if history:
            history.trim_to_fit(self.max_tokens, self.tokenizer)

        response = self.llm.chat(messages, system_prompt=system_prompt)
        if history:
            history.add("assistant", response)
        try:
            return json.loads(response)
        except Exception as e:
            return {"error": f"Failed to parse response as JSON: {str(e)}", "raw_response": response}

    def run_and_time_kernels(
        self, 
        kernels_config,
        backend="cuda", 
        matrix_dim=4096
    ):
        """
        kernels_config: output of generate_kernels (dict with key 'kernels', each a dict)
        """
        results = []
        for kernel in kernels_config["kernels"]:
            kernel_code = kernel["kernel_code"]
            kernel_name = kernel.get("kernel_name", None)
            global_size = kernel["global_size"]
            local_size = kernel["local_size"]
            input_setup_code = kernel["input_setup_code"]
            verification_code = kernel["verification_code"]

            # Dynamically create input_setup_fn and verification_fn from code strings
            input_setup_fn = self._make_callable_from_code(input_setup_code, "input_setup")
            verification_fn = self._make_callable_from_code(verification_code, "verify")

            if backend == "cuda":
                runner = self.cuda_runner
                result = runner.run_kernel(
                    kernel_code=kernel_code,
                    kernel_name=kernel_name,
                    input_setup_fn=lambda: input_setup_fn(runner, matrix_dim),
                    verification_fn=lambda: verification_fn(runner, matrix_dim),
                    num_iterations=5
                )
            elif backend == "opencl":
                runner = self.opencl_runner
                result = runner.run_kernel(
                    kernel_code=kernel_code,
                    kernel_name=kernel_name,
                    input_setup_fn=lambda ctx: input_setup_fn(runner, matrix_dim, ctx),
                    verification_fn=lambda: verification_fn(runner, matrix_dim),
                    num_iterations=5,
                    block_size=local_size,
                    grid_size=global_size
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
            results.append(result)
        return results

    def _make_callable_from_code(self, code_str, func_name):
        """
        Given code_str defining a function named func_name, return the function object.
        """
        local_vars = {}
        exec(code_str, globals(), local_vars)
        return local_vars[func_name]