#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;
constant int BLOCK_SIZE = 64;

kernel void matmul4096(device const float* A  [[buffer(0)]],
                       device const float* B  [[buffer(1)]],
                       device float* C  [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]])
{
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = gid.y;
    int col = gid.x;
    float sum = 0.0;
    
    for(int t=0; t<4096; t+=TILE_SIZE) {
        // Load tiles into shared memory
        for(int i=0; i<TILE_SIZE; i+=BLOCK_SIZE) {
            tileA[i+row%TILE_SIZE][col%TILE_SIZE] = A[row*4096 + (t+i+col%TILE_SIZE)];
            tileB[i+row%TILE_SIZE][col%TILE_SIZE] = B[(t+i+row%TILE_SIZE)*4096 + col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int k=0; k<TILE_SIZE; k++) {
            sum += tileA[row%TILE_SIZE][k] * tileB[k][col%TILE_SIZE];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    C[row*4096 + col] = sum;
}