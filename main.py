import json
import os
from dotenv import load_dotenv

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.openai_provider import OpenAIProvider
from src.prompt.history import PromptHistory
from src.gpu_types import GPUType
from src.kernel.kernel_manager import KernelManager
from src.utils.device_info import get_device_info

load_dotenv()

def simple_tokenizer(messages):
    return sum(len(m["content"].split()) for m in messages)

# LLM provider selection
provider = os.getenv("LLM_PROVIDER", "anthropic")
if provider == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAIProvider(openai_api_key, model="o4-mini")
else:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    llm = AnthropicProvider(anthropic_api_key)

# Kernel manager setup
kernel_manager = KernelManager(llm, simple_tokenizer, max_tokens=8000)

# Tasks
with open("data/tasks.json", "r", encoding="utf-8") as f:
    tasks_data = json.load(f)

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":
    backend = os.environ.get("AUTOKERNEL_BACKEND", "opencl")
    gpu_type = GPUType.Nvidia if backend == "cuda" else GPUType.Qualcomm
    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software

    device_info = get_device_info(backend)
    print("\nDevice Information:")
    device_info_str = ""
    for key, value in device_info.items():
        device_info_str += f"{key}: {value}\n"

    if "tasks" in tasks_data:
        tasks_list = tasks_data["tasks"]
        for task_entry in tasks_list:
            task = task_entry["task"].format(
                gpu_software=gpu_software,
                gpu_manufacturer=gpu_manufacturer,
                gpu_hardware=gpu_hardware
            )
            timing_data = None
            kernel_config = None
            output_filename = f"output/generated_kernel.{gpu_software.lower()}"
            kernel_configs = []
            history = PromptHistory()
            for i in range(5):
                constraints = (
                    "Minimize global memory reads and writes. "
                    + f"Maximize usage of {gpu_hardware} compute units without exceeding register limits. "
                    + "Minimize the processing time on the GPU. "
                    + "Maximize GPU utilization. "
                    + f"Device Information: {device_info_str}"
                    + (
                        f"\nThe previous generated kernel: {kernel_config} took {timing_data['average_ms']} ms. Optimize further keeping in mind the device info such as max work group size and memory limitations."
                        if timing_data and 'average_ms' in timing_data
                        else (
                            f"\nThe previous generated kernel: {kernel_config} failed to run. Error: {timing_data['error']}. Please fix the issue and optimize further."
                            if timing_data and 'error' in timing_data
                            else ""
                        )
                    )
                )

                manual_context = None

                kernel_config = kernel_manager.generate_kernel(
                    gpu_type, task, constraints, manual_context, history
                )
                timing_data = kernel_manager.run_and_time_kernel(kernel_config, backend=backend)
                kernel_configs.append(kernel_config)
            history.save("output/prompt_history.json")
            with open(output_filename, "w", encoding='utf-8') as f:
                f.write(json.dumps(kernel_configs, indent=4, ensure_ascii=False))