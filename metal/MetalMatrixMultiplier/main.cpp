#include <iostream>
#include <chrono>
#include <cmath>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalMatrixMultiplier.hpp"

// Typedef for timing
typedef std::chrono::microseconds time_unit;
const char* unit_name = "microseconds";

int main()
{
    // Create Metal device
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device)
    {
        std::cerr << "Failed to create Metal device!" << std::endl;
        return -1;
    }

    // Create matrix multiplier
    MetalMatrixMultiplier *matmul = new MetalMatrixMultiplier(device);

    // Warm-up: run once to compile Metal shader
    matmul->sendComputeCommand();
    matmul->verifyResults();

    // Benchmarking parameters
    const int repeats = 20;
    auto durations = new float[repeats];

    // Benchmark GPU matrix multiplication
    for (int i = 0; i < repeats; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        matmul->sendComputeCommand();
        auto stop = std::chrono::high_resolution_clock::now();
        durations[i] = std::chrono::duration_cast<time_unit>(stop - start).count();
    }

    // Compute mean and standard deviation
    float mean = 0.0f;
    for (int i = 0; i < repeats; ++i) mean += durations[i];
    mean /= repeats;

    float variance = 0.0f;
    for (int i = 0; i < repeats; ++i) variance += (durations[i] - mean) * (durations[i] - mean);
    float stddev = std::sqrt(variance / repeats);

    // Print performance results
    std::cout << "Metal GPU Matrix Multiply:" << std::endl;
    std::cout << mean << " " << unit_name << " ± " << stddev << " " << unit_name << std::endl;

    // Compute TFLOPS
    double ops = 2.0 * M * N * K; // 2 * M * N * K for GEMM
    double tflops = ops / (mean * 1e-6) / 1e12;
    std::cout << "Approximate TFLOPS: " << tflops << " TFLOPS" << std::endl;

    // Clean up
    delete[] durations;
    delete matmul;
    device->release();

    return 0;
}
