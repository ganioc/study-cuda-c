#include <iostream>
#include <math.h>
#include <stdio.h>

#define STRIDE_64K  65536


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
    int device = -1;
    // Allocate Unified Memoy, from 
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for(int i = 0; i< N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);
    
    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize -1)/ blockSize;
//    init<<<numBlocks, blockSize>>>(N, x, y);
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

