import json
import os
from dotenv import load_dotenv

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.openai_provider import OpenAIProvider
from src.prompt.history import PromptHistory
from src.gpu_types import GPUType
from src.kernel.kernel_manager import KernelManager
from src.utils.device_info import get_device_info
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
tasks_data = None

if os.path.exists("data/task_sakana.json"):
    with open("data/task_sakana.json", "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Manifest
best_kernel_files = []

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
        for task_entry in tasks_list[:10]:
            if task_entry["disabled"]:
                print(f"Skipping disabled task: {task_entry['kernel_name']}")
                continue
            if "code" in task_entry and task_entry["code"]:                
                task = task_entry["task"].format(
                    gpu_software=gpu_software,
                    gpu_manufacturer=gpu_manufacturer,
                    gpu_hardware=gpu_hardware,
                    code=task_entry["PyTorch_Code_Module"],
                    cuda_sample=task_entry["code"]
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
            kernel_configs = {}
            history = PromptHistory()

            for i in range(3):
                constraints = (
                    "Minimize global memory reads and writes. "
                    + f"Maximize usage of {gpu_hardware} compute units without exceeding register limits. "
                    + "Minimize the processing time on the GPU. "
                    + "Maximize GPU utilization. "
                    + f"Device Information: {device_info_str}"
                    + (
                        f"\nThe previous generated kernel: {kernel_configs[i-1]['kernel_code']} took {kernel_configs[i-1]['timing_info']['average_ms']} ms. The native torch implementation takes {kernel_configs[i-1]['torch_timing_info']['average_ms']} ms. Optimize to beat the native torch implementation keeping in mind the device info such as max work group size and memory limitations."
                        if i > 0 and kernel_configs.get(i-1) and kernel_configs[i-1]['timing_info'] and kernel_configs[i-1]['torch_timing_info'] and 'average_ms' in kernel_configs[i-1]['timing_info'] and 'average_ms' in kernel_configs[i-1]['torch_timing_info']
                        else (
                            f"\nThe previous generated kernel: {kernel_configs[i-1]['kernel_code']} failed to run. Error: {kernel_configs[i-1]['timing_info']['error']}. Please fix the issue and optimize further."
                            if i > 0 and kernel_configs.get(i-1) and kernel_configs[i-1]['timing_info'] and 'error' in kernel_configs[i-1]['timing_info']
                            else ""
                        )
                    )
                )

                manual_context = None

                kernel_config = kernel_manager.generate_kernels(
                    gpu_type, task, constraints, manual_context, history
                )

                if 'kernels' not in kernel_config:
                    continue
                
                # Run torch kernel for comparision
                print(f"Running native torch for kernel id 0")
                local_vars_torch = {}
                exec(kernel_config['kernels'][0]['native_torch_setup'], globals(), local_vars_torch)
                native_torch_setup_fn = local_vars_torch['native_torch_setup']
                torch_timing_data = None
                try:
                    torch_timing_data = run_with_timeout(native_torch_setup_fn, timeout=20)  # 10 seconds timeout
                    if torch_timing_data is None:
                        raise Exception("Torch function did not complete in time.")
                    else:
                        print(f"Torch kernel took {torch_timing_data['average_ms']} ms")
                except Exception as e:
                    import traceback
                    print(e)
                    print(traceback.format_exc())
                    torch_timing_data = {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }

                for kernel_id in range(len(kernel_config['kernels'])):
                    print(f"Running kernel {kernel_id} for iteration {i}")
                    local_vars = {}
                    exec(kernel_config['kernels'][kernel_id]['runner_setup'], globals(), local_vars)
                    runner_setup_fn = local_vars['runner_setup']
                    try:
                        timing_data = run_with_timeout(runner_setup_fn, timeout=20)  # 10 seconds timeout
                        if timing_data is None:
                            raise Exception("Setup function did not complete in time.")
                        kernel_configs[i] = {
                            "kernel_code": kernel_config['kernels'][kernel_id]['runner_setup'],
                            "timing_info": timing_data,
                            "torch_timing_info": torch_timing_data
                        }
                    except Exception as e:
                        import traceback
                        print(e)
                        print(traceback.format_exc())
                        timing_data = {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                        kernel_configs[i] = {
                            "kernel_code": kernel_config['kernels'][kernel_id]['runner_setup'],
                            "timing_info": timing_data,
                            "torch_timing_info": torch_timing_data
                        }

            print("-------------------------------------------------------------------------------------")
            history.save(f"output/prompt_history_{task_entry['kernel_name']}.json")
            kernel_and_timing_strings = []
            for i in range(len(kernel_configs.keys())):
                if i in kernel_configs:
                    kernel_str = str(kernel_configs[i]['kernel_code'])
                    try:
                        timing_info = str(json.dumps(kernel_configs[i]['timing_info'], indent=2))
                    except TypeError:
                        # Fallback: convert non-serializable objects to string
                        def default_serializer(obj):
                            try:
                                return str(obj)
                            except Exception:
                                return "<unserializable>"
                        timing_info = str(json.dumps(kernel_configs[i]['timing_info'], indent=2, default=default_serializer))
                    kernel_and_timing_strings.append(f"{kernel_str}\n{timing_info}")

            # Save kernel_and_timing_strings in a text file
            with open(output_filename, "w", encoding='utf-8') as txt_f:
                txt_f.write(task)
                txt_f.write("\n\n-------------------------------------------\n\n")
                txt_f.write("\n\n".join(kernel_and_timing_strings))
            # After collecting kernel_and_timing_strings, also save the best kernel as JSON
            best_idx = min(
                (i for i in kernel_configs if 'average_ms' in kernel_configs[i]['timing_info'] and 'correct_result' in kernel_configs[i]['timing_info'] and kernel_configs[i]['timing_info']['correct_result']),
                key=lambda i: kernel_configs[i]['timing_info']['average_ms'],
                default=None
            )
            if best_idx is not None:
                best_kernel = {
                    "task": task,
                    "task_name": task_entry['kernel_name'],
                    "kernel_code": kernel_configs[best_idx]['kernel_code'],
                    "timing_info": kernel_configs[best_idx]['timing_info']
                }
                best_kernel_filename = f"docs/best_kernel_{task_entry['kernel_name']}.json"
                def default_serializer(obj):
                    try:
                        return str(obj)
                    except Exception:
                        return "<unserializable>"
                with open(best_kernel_filename, "w", encoding='utf-8') as jf:
                    json.dump(best_kernel, jf, indent=2, default=default_serializer)          
                best_kernel_files.append(os.path.basename(best_kernel_filename))
        manifest_path = "docs/manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(best_kernel_files, mf, indent=2)