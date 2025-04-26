__kernel void matrixMul(__global float* A, __global float* B, __global float* C, int N) {
    
    __local float Asub[16][16];
    __local float Bsub[16][16];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    
    int row = by*16 + ty;
    int col = bx*16 + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < N/16; ++m) {
        Asub[ty][tx] = A[row*N + (m*16 + tx)];
        Bsub[ty][tx] = B[(m*16 + ty)*N + col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < 16; ++k) {
            sum += Asub[ty][k] * Bsub[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[row*N + col] = sum;
}