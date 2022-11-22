#include <iostream>
#include <math.h>
#include <stdio.h>

#define STRIDE_64K  65536


__global__ void init(int n, float *x, float *y){
    int lane_id = threadIdx.x &31;
    size_t warp_id = (threadIdx.x + blockDim.x * blockIdx.x)>> 5;
    size_t warps_per_grid = (blockDim.x* gridDim.x) >>5;
    size_t warp_total = ((sizeof(float) *n) + STRIDE_64K-1) / STRIDE_64K;


//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    for(int i=index; i< n; i+= stride){
//        x[i] = 1.0f;
//        y[i] = 2.0f;
//    }
    for(; warp_id < warp_total; warp_id += warps_per_grid){
        #pragma unroll
        for(int rep = 0; rep < STRIDE_64K/sizeof(float)/32; rep++){
            size_t ind = warp_id * STRIDE_64K/sizeof(float) + rep*32 + lane_id;
            if( ind < n){
                x[ind] = 1.0f;
                y[ind] = 2.0f;
            }
        }
    }
}

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i< n; i+= stride){
        y[i] = x[i] + y[i];
    }
}
int main(void){
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memoy, from 
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
//    for(int i = 0; i< N; i++){
  //      x[i] = 1.0f;
  //      y[i] = 2.0f;
  //  }
    
    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize -1)/ blockSize;
    init<<<numBlocks, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values shouold be 3.0f)
    float maxError = 0.0f;
    for(int i = 0; i< N; i++){
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout <<"Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    return 0;

}


