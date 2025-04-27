#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

// Define matrix dimensions
const unsigned int M = 4096; // Rows of A and C
const unsigned int N = 4096; // Columns of B and C
const unsigned int K = 4096; // Columns of A, Rows of B

const unsigned int bufferSizeA = M * K * sizeof(float);
const unsigned int bufferSizeB = K * N * sizeof(float);
const unsigned int bufferSizeC = M * N * sizeof(float);

class MetalMatrixMultiplier
{
public:
    MTL::Device *_mDevice;
    MTL::ComputePipelineState *_mMatMulFunctionPSO;
    MTL::CommandQueue *_mCommandQueue;

    MTL::Buffer *_mBufferA;
    MTL::Buffer *_mBufferB;
    MTL::Buffer *_mBufferC;

    MetalMatrixMultiplier(MTL::Device *device);
    ~MetalMatrixMultiplier();

    void prepareData();
    void sendComputeCommand();
    void verifyResults();

private:
    void encodeMatMulCommand(MTL::ComputeCommandEncoder *computeEncoder);
    void generateRandomFloatData(MTL::Buffer *buffer, size_t count);
};
