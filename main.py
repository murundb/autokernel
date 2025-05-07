import json
import os
from dotenv import load_dotenv

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.openai_provider import OpenAIProvider
from src.prompt.history import PromptHistory
from src.gpu_types import GPUType
from src.kernel.kernel_manager import KernelManager
from src.utils.device_info import get_device_info
from src.utils.dynamic_setup import create_input_setup_fn, create_verification_fn
import concurrent.futures

def run_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Function timed out after {timeout} seconds")
            return None
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
            if task_entry["disabled"]:
                print(f"Skipping disabled task: {task_entry['kernel_name']}")
                continue
            if task_entry["code"]:
                task = task_entry["task"].format(
                    gpu_software=gpu_software,
                    gpu_manufacturer=gpu_manufacturer,
                    gpu_hardware=gpu_hardware,
                    code=task_entry["code"]
                )
            else:
                task = task_entry["task"].format(
                    gpu_software=gpu_software,
                    gpu_manufacturer=gpu_manufacturer,
                    gpu_hardware=gpu_hardware
                )
            print(f"Task: {task}")
            timing_data = None
            kernel_config = None
            output_filename = f"output/generated_kernel_{task_entry['kernel_name']}.txt"
            kernel_configs = []
            history = PromptHistory()

            for i in range(1):
                constraints = (
                    "Minimize global memory reads and writes. "
                    + f"Maximize usage of {gpu_hardware} compute units without exceeding register limits. "
                    + "Minimize the processing time on the GPU. "
                    + "Maximize GPU utilization. "
                    + f"Device Information: {device_info_str}"
                    + (
                        f"\nThe previous generated kernel: {kernel_configs[-1]} took {timing_data['average_ms']} ms. Optimize further keeping in mind the device info such as max work group size and memory limitations."
                        if timing_data and 'average_ms' in timing_data
                        else (
                            f"\nThe previous generated kernel: {kernel_configs[-1]} failed to run. Error: {timing_data['error']}. Please fix the issue and optimize further."
                            if timing_data and 'error' in timing_data
                            else ""
                        )
                    )
                )

                manual_context = None

                kernel_config = kernel_manager.generate_kernels(
                    gpu_type, task, constraints, manual_context, history
                )

                for kernel_id in range(len(kernel_config['kernels'])):
                    print(f"Running kernel {kernel_id} for iteration {i}")
                    local_vars = {}
                    exec(kernel_config['kernels'][kernel_id]['runner_setup'], globals(), local_vars)
                    kernel_configs.append(kernel_config['kernels'][kernel_id]['runner_setup'])
                    runner_setup_fn = local_vars['runner_setup']
                    try:
                        timing_data = run_with_timeout(runner_setup_fn, timeout=20)  # 10 seconds timeout
                        if timing_data is None:
                            raise Exception("Setup function did not complete in time.")
                        kernel_configs.append(timing_data)
                    except Exception as e:
                        import traceback
                        print(e)
                        print(traceback.format_exc())
                        timing_data = {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                        kernel_configs.append(timing_data)

            print("-------------------------------------------------------------------------------------")
            history.save(f"output/prompt_history_{task_entry['kernel_name']}.json")
            kernel_and_timing_strings = []
            for idx in range(0, len(kernel_configs), 2):
                kernel_str = str(kernel_configs[idx])
                timing_info = str(json.dumps(kernel_configs[idx + 1], indent=2)) if idx + 1 < len(kernel_configs) else ""
                kernel_and_timing_strings.append(f"{kernel_str}\n{timing_info}")

            # Save kernel_and_timing_strings in a text file
            with open(output_filename, "w", encoding='utf-8') as txt_f:
                txt_f.write(task)
                txt_f.write("\n")
                txt_f.write("\n\n".join(kernel_and_timing_strings))