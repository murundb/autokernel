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

    def run_and_time_kernel(
        self, 
        kernel_name,
        kernel_config,
        matrix_dim=4096, 
        backend="cuda", 
        input_setup_fn=None, 
        verification_fn=None
    ):
        try:
            if backend == "cuda":
                runner = self.cuda_runner
                return runner.run_kernel(
                    kernel_code=kernel_config["kernel_code"],
                    kernel_name=kernel_name,
                    input_setup_fn=lambda: input_setup_fn(runner, matrix_dim),
                    verification_fn=lambda: verification_fn(runner, matrix_dim),
                    num_iterations=5
                )
            elif backend == "opencl":
                runner = self.opencl_runner
                return runner.run_kernel(
                    kernel_code=kernel_config["kernel_code"],
                    kernel_name=kernel_name,
                    input_setup_fn=lambda ctx: input_setup_fn(runner, matrix_dim, ctx),
                    verification_fn=lambda: verification_fn(runner, matrix_dim),
                    num_iterations=5,
                    block_size=kernel_config["local_size"],
                    grid_size=kernel_config["global_size"]
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }