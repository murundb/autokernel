import os
import openai
from agents import Agent, Runner, ModelSettings, function_tool
from dotenv import load_dotenv

from src.gpu_types import GPUType

load_dotenv()  # take environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_NAME = "KernelAgent"
OUTPUT_PATH = "generated_kernel.metal"

OUTPUT_KERNEL_FOLDER_PATH = "output/generated_kernels"
OUTPUT_METADATA_FOLDER_PATH = "output/metadata"

def save_kernel(kernel_text, generation_number):
    filename = "generated_kernel_{}.metal".format(generation_number)
    output_fqp = os.path.join(OUTPUT_KERNEL_FOLDER_PATH, filename)

    with open(output_fqp, "w") as f:
        f.write(kernel_text)
    
    print("Saved kernel to {}".format(filename))

def save_metadata(task, generation_number, instructions=None, feedback=None):

    filename = "metadata_kernel_{}.log".format(generation_number)
    output_fqp = os.path.join(OUTPUT_METADATA_FOLDER_PATH, filename)
    with open(output_fqp, "w") as f:

        if (instructions):
            f.write("Instructions:\n")
            f.write(instructions)
            f.write("\n")

        if (feedback):
            f.write("Feedback:\n")
            f.write(feedback)
            f.write("\n")
        
        f.write("Task:\n")
        f.write(task)
        f.write("\n")

    print("Saved metadata to {}".format(filename))

def main():

    if not os.path.exists(OUTPUT_KERNEL_FOLDER_PATH):
        os.makedirs(OUTPUT_KERNEL_FOLDER_PATH)

    if not os.path.exists(OUTPUT_METADATA_FOLDER_PATH):
        os.makedirs(OUTPUT_METADATA_FOLDER_PATH)

    gpu_type = GPUType.Apple
    gpu_manufacturer = gpu_type.name
    gpu_hardware = gpu_type.value.hardware
    gpu_software = gpu_type.value.software


    ## == Initialize agent == 
    # This is system prompt
    instructions = "You are an expert {0} {1} GPU engineer. Your job is correctness and holding to the given task specification. Your main task is to generate highly efficient {2} kernels for {3} {4} GPUs. Make sure the kernel returns the correct result. Do no add markdown, explanations, comments, or anything else-only code. Do not use any alternative precision that could result in an incorrect result. Focus on minimizing execution speed, minimizing global memory access, maximizing parallel execution, optimizing register usage, and leveraging local/shared memory. If user provides feedback, improve correctness and performance based on the feedback.".format(gpu_manufacturer, gpu_hardware, gpu_software, gpu_manufacturer, gpu_hardware)

    agent = Agent(name=AGENT_NAME, 
                  handoff_description="Specialist agent for kernel generation",
                  instructions=instructions,
                  model="gpt-4o",
                  tools=[])
    
    ## == Generate and save the initial kernel and initial metadata == 
    cnt_gen = 0
    init_task = "Generate a basic {0} kernel with a function name called matmul that multiplies two 4096x4096 matrices A and B into C. ".format(gpu_hardware)
    result = Runner.run_sync(agent, init_task)

    save_kernel(result.final_output, cnt_gen)
    save_metadata(init_task, cnt_gen, instructions=instructions)

    while True:
        print("Generation {}".format(cnt_gen))

        feedback_lines = []

        try:
            while True:
                line = input()
                feedback_lines.append(line)
        except EOFError:
            pass

        feedback_text = "\n".join(feedback_lines)

        print("Feedback is: ")
        print(feedback_text)

        task = f"Feedback:\n{feedback_text}\n\nPlease generate an improved full Metal kernel matmul based on this feedback. Make sure the kernel returns the correct result. Do no add markdown, explanations, comments, or anything else-only code."

        result = Runner.run_sync(agent, task)

        cnt_gen += 1
        save_kernel(result.final_output, cnt_gen)
        save_metadata(task, cnt_gen)
if __name__ == "__main__":
    main()