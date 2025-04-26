import anthropic
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables

# Load your API key securely
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def generate_kernel(task_description, constraints=None, manual_context=None):
    system_prompt = (
        "You are an expert Qualcomm Adreno GPU engineer. "
        "Your task is to generate highly efficient OpenCL kernels for Qualcomm Adreno GPUs. "
        "Focus on minimizing global memory access, minimizing runtime, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. "
        "Use precise, hardware-conscious code generation, following Qualcomm's architectural best practices."
    )

    user_prompt = f"""
Refer to the following Qualcomm Adreno architecture context (extracted from the manual):

{adreno_manual_context if adreno_manual_context else "No additional context provided."}

---

Target Task:
{task_description}

Constraints:
{constraints if constraints else "No specific constraints provided."}

---

Instructions:
- Output a fully working OpenCL or CUDA kernel.
- Focus on optimizing for Qualcomm Adreno GPUs.
- Return only the code, no explanations, no markdown formatting, no extra commentary.
"""

    response = client.messages.create(
        model="claude-3-opus-20240229",   # You can change model if you want (haiku, sonnet, opus)
        max_tokens=2000,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    generated_kernel_code = response.content[0].text

    return generated_kernel_code

if __name__ == "__main__":

    task = "Write an OpenCL kernel that performs 512x512 matrix multiplication optimized for Qualcomm Adreno architecture."

    constraints = (
        "Use local memory tiling with 16x16 tiles. "
        "Minimize global memory reads and writes. "
        "Maximize usage of Adreno compute units without exceeding register limits. "
        "Assume OpenCL 2.0 availability."
        "Minimize the processing time on the GPU."
        "Maximize GPU utilization"
    )

    adreno_manual_context = """
- Global memory accesses are slow; prefer coalesced reads/writes.
- Use __local memory for intermediate computations.
- Synchronization using barrier(CLK_LOCAL_MEM_FENCE) is available.
- Prefer vectorized loads (e.g., float4) when accessing memory.
- Tile sizes of 16x16 or 32x32 usually map well to Adreno thread dispatch units.
- Avoid excessive branching inside kernels.
"""

    kernel_code = generate_kernel(task, constraints, adreno_manual_context)

    # Save the generated kernel to a file
    output_filename = "generated_kernel.cl"
    with open(output_filename, "w") as f:
        f.write(kernel_code)