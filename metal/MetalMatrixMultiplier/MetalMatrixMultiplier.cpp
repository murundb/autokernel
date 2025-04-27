#include "MetalMatrixMultiplier.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

MetalMatrixMultiplier::MetalMatrixMultiplier(MTL::Device *device)
{
    _mDevice = device;
    NS::Error *error = nullptr;

    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
    if (!defaultLibrary)
    {
        std::cout << "Failed to load default library." << std::endl;
        return;
    }

    MTL::Function *matMulFunction = defaultLibrary->newFunction(
        NS::String::string("matmul", NS::ASCIIStringEncoding));
    defaultLibrary->release();

    if (!matMulFunction)
    {
        std::cout << "Failed to find matmul function." << std::endl;
        return;
    }

    _mMatMulFunctionPSO = _mDevice->newComputePipelineState(matMulFunction, &error);
    matMulFunction->release();

    if (!_mMatMulFunctionPSO)
    {
        std::cout << "Failed to create pipeline state: " << error << std::endl;
        return;
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (!_mCommandQueue)
    {
        std::cout << "Failed to create command queue." << std::endl;
        return;
    }

    _mBufferA = _mDevice->newBuffer(bufferSizeA, MTL::ResourceStorageModeShared);
    _mBufferB = _mDevice->newBuffer(bufferSizeB, MTL::ResourceStorageModeShared);
    _mBufferC = _mDevice->newBuffer(bufferSizeC, MTL::ResourceStorageModeShared);

    prepareData();
}

void MetalMatrixMultiplier::prepareData()
{
    generateRandomFloatData(_mBufferA, M * K);
    generateRandomFloatData(_mBufferB, K * N);
}

void MetalMatrixMultiplier::sendComputeCommand()
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer);

    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder);

    encodeMatMulCommand(computeEncoder);

    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalMatrixMultiplier::encodeMatMulCommand(MTL::ComputeCommandEncoder *computeEncoder)
{
    computeEncoder->setComputePipelineState(_mMatMulFunctionPSO);
    computeEncoder->setBuffer(_mBufferA, 0, 0);
    computeEncoder->setBuffer(_mBufferB, 0, 1);
    computeEncoder->setBuffer(_mBufferC, 0, 2);

    computeEncoder->setBytes(&M, sizeof(unsigned int), 3);
    computeEncoder->setBytes(&N, sizeof(unsigned int), 4);
    computeEncoder->setBytes(&K, sizeof(unsigned int), 5);

    MTL::Size gridSize = MTL::Size::Make(N, M, 1);
    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1); // reasonable (tune if you want)

    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void MetalMatrixMultiplier::generateRandomFloatData(MTL::Buffer *buffer, size_t count)
{
    float *data = reinterpret_cast<float *>(buffer->contents());
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void MetalMatrixMultiplier::verifyResults()
{
    float *a = reinterpret_cast<float *>(_mBufferA->contents());
    float *b = reinterpret_cast<float *>(_mBufferB->contents());
    float *c = reinterpret_cast<float *>(_mBufferC->contents());

    for (unsigned int row = 0; row < M; ++row)
    {
        for (unsigned int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (unsigned int k = 0; k < K; ++k)
            {
                sum += a[row * K + k] * b[k * N + col];
            }
            float diff = fabs(sum - c[row * N + col]);
            if (diff > 1e-3f)
            {
                printf("Mismatch at (%u, %u): got %f, expected %f\n", row, col, c[row * N + col], sum);
                assert(false);
            }
        }
    }
}

MetalMatrixMultiplier::~MetalMatrixMultiplier()
{
    _mBufferA->release();
    _mBufferB->release();
    _mBufferC->release();
    _mMatMulFunctionPSO->release();
    _mCommandQueue->release();
}
