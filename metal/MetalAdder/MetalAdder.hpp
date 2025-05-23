#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

// The number of floats in each array, and the size of the arrays in bytes.
const unsigned int arrayLength = 60 * 180 * 10000;

const unsigned int bufferSize = arrayLength * sizeof(float);

class MetalAdder
{
public:
    MTL::Device *_mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    MTL::ComputePipelineState *_mAddFunctionPSO;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;

    // Buffers to hold data.
    MTL::Buffer *_mBufferA;
    MTL::Buffer *_mBufferB;
    MTL::Buffer *_mBufferResult;

    MetalAdder(MTL::Device *device);
    ~MetalAdder();

    void prepareData();
    void sendComputeCommand();
    void verifyResults();

private:
    void encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder);
    void generateRandomFloatData(MTL::Buffer *buffer);
};