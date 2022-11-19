#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
}

int main(void){
    printf("Hello from host\n");
    print_from_gpu<<<2,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
