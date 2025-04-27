import torch
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Matrix size
size = 4096  # You can try 8192 but it will use a lot of memory (~2 GB for float32)

# Create two random matrices on the GPU
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

# Warm-up run (important for fair timing)
_ = torch.matmul(A, B)
torch.mps.synchronize()

# Time the real multiplication
start = time.time()
C = torch.matmul(A, B)
torch.mps.synchronize()  # Ensure GPU work is completed
end = time.time()

elapsed_time = end - start

# Print results
print(f"Matrix multiplication size: {size}x{size}")
print(f"Time taken: {elapsed_time:.4f} seconds")

# Optional: Compute theoretical FLOPs
num_flops = 2 * size**3  # 2 * m * n * k for GEMM
tflops = num_flops / (elapsed_time * 1e12)
print(f"Performance: {tflops:.2f} TFLOPs")