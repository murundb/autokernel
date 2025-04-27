import anthropic
import os
from dotenv import load_dotenv

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
        "Focus on minimizing execution speed, minimizing global memory access, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. "
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
    - Output a fully working {gpu_software} kernel.
    - Focus on optimizing for {gpu_manufacturer} {gpu_hardware} GPUs.
    - Return only the code, no explanations, no markdown formatting, no extra commentary.
    """

    response = client.messages.create(
        model="claude-3-opus-20240229",   # You can change model if you want (haiku, sonnet, opus)
        max_tokens=2000,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {"role": "user", "content": base_prompt}
        ]
    )

    generated_kernel_code = response.content[0].text

    return generated_kernel_code

if __name__ == "__main__":

    gpu_type = GPUType.Apple
    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software

    task = "Write {0} kernel that performs 4096x4096 matrix multiplication optimized for {1} {2} architecture.".format(gpu_software, gpu_manufacturer, gpu_hardware)

    # constraints = (
    #     "Use local memory tiling with 16x16 tiles. "
    #     "Minimize global memory reads and writes. "
    #     "Maximize usage of Adreno compute units without exceeding register limits. "
    #     "Assume OpenCL 2.0 availability."
    #     "Minimize the processing time on the GPU."
    #     "Maximize GPU utilization"
    # )

#     manual_context = """
# - Global memory accesses are slow; prefer coalesced reads/writes.
# - Use __local memory for intermediate computations.
# - Synchronization using barrier(CLK_LOCAL_MEM_FENCE) is available.
# - Prefer vectorized loads (e.g., float4) when accessing memory.
# - Tile sizes of 16x16 or 32x32 usually map well to Adreno thread dispatch units.
# - Avoid excessive branching inside kernels.
# """

    constraints = (
        "Minimize global memory reads and writes. "
        + "Maximize usage of {} compute units without exceeding register limits. ".format(gpu_hardware)
        + "Minimize the processing time on the GPU. "
        + "Maximize GPU utilization."
    )

    manual_context = None

    kernel_code = generate_kernel(gpu_type, task, constraints, manual_context)

    # Save the generated kernel to a file
    output_filename = "generated_kernel.metal"
    with open(output_filename, "w") as f:
        
        f.write(kernel_code)