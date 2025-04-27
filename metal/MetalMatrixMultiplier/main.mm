// This is benchmark for MPSMatrixMultiplication

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>

#include <iostream>
#include <chrono>
#include <cmath>

// Your matrix sizes
const unsigned int M = 4096;
const unsigned int N = 4096;
const unsigned int K = 4096;

// Typedef for timing
typedef std::chrono::microseconds time_unit;
const char* unit_name = "microseconds";

int main()
{
    @autoreleasepool
    {
        // Create Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            std::cerr << "Failed to create Metal device!" << std::endl;
            return -1;
        }

        // Create Metal buffers
        id<MTLBuffer> bufferA = [device newBufferWithLength:M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];

        // Fill A and B with random data
        float* ptrA = (float*)bufferA.contents;
        float* ptrB = (float*)bufferB.contents;
        for (unsigned int i = 0; i < M * K; i++) ptrA[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        for (unsigned int i = 0; i < K * N; i++) ptrB[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        // Create MPS descriptors
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        // Wrap Metal buffers into MPSMatrix
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        // Create MPSMatrixMultiplication
        MPSMatrixMultiplication* mpsGEMM = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                   transposeLeft:false
                                                                  transposeRight:false
                                                                            resultRows:M
                                                                         resultColumns:N
                                                                         interiorColumns:K
                                                                                  alpha:1.0
                                                                                   beta:0.0];

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // Warm-up run
        {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            [mpsGEMM encodeToCommandBuffer:commandBuffer
                                   leftMatrix:matrixA
                                  rightMatrix:matrixB
                                      resultMatrix:matrixC];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        // Benchmark
        const int repeats = 20;
        auto durations = new float[repeats];

        for (int i = 0; i < repeats; ++i)
        {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            auto start = std::chrono::high_resolution_clock::now();

            [mpsGEMM encodeToCommandBuffer:commandBuffer
                                  leftMatrix:matrixA
                                 rightMatrix:matrixB
                                 resultMatrix:matrixC];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            auto stop = std::chrono::high_resolution_clock::now();
            durations[i] = std::chrono::duration_cast<time_unit>(stop - start).count();
        }

        // Statistics
        float mean = 0.0f;
        for (int i = 0; i < repeats; ++i) mean += durations[i];
        mean /= repeats;

        float variance = 0.0f;
        for (int i = 0; i < repeats; ++i) variance += (durations[i] - mean) * (durations[i] - mean);
        float stddev = std::sqrt(variance / repeats);

        std::cout << "MPSMatrixMultiplication performance:" << std::endl;
        std::cout << mean << " " << unit_name << " ± " << stddev << " " << unit_name << std::endl;

        // Compute TFLOPS
        double ops = 2.0 * M * N * K;
        double tflops = ops / (mean * 1e-6) / 1e12;
        std::cout << "Approximate TFLOPS: " << tflops << " TFLOPS" << std::endl;

        delete[] durations;
    }

    return 0;
}
