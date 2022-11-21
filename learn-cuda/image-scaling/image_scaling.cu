#include<stdio.h>
#include"imgPgm.h"

// Kernel which calculate the resized image,
__global__ void createResizedImage(unsigned char *imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj){
    const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned index = tidY * scaled_width + tidX;

    // Read the texture memory from your texture reference in CUDA Kernel
    imageScaledData[index] = tex2D<unsigned char>(texObj, (float) (tidX*scale_factor), (float)(tidY*scale_factor));
}

int main(int argc,char**argv){
    int height = 0, width = 0, scaled_height = 0, scaled_width = 0;
    // Define the scaling ratio,
    float scaling_ratio = 0.5;
    unsigned char* data;
    unsigned char* scaled_data, *d_scaled_data;

    char inputStr[1024] = {"aerosmith-double.pgm"};
    char outputStr[1024] = {"aerosmith-double-scaled.pgm"};
    cudaError_t returnValue;

    // Create a channel Description to be used while linking to the texture
    cudaArray* cu_array;
    cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,kind);

    get_PgmPpmParams(inputStr, &height, &width);
    data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
    printf("\n Reading image height and width [%d] [%d]\r\n", height , width);
    src_read_pgm(inputStr, data, height, width);
    // height is rows, widht is cols, it's right!
    // loading an image into data,

    scaled_height = (int)(height * scaling_ratio);
    scaled_width = (int)(width * scaling_ratio);
    scaled_data = (unsigned char*)malloc(scaled_height * scaled_width * sizeof(unsigned char));
    printf("\n Scaled image height and width [%d] [%d]\r\n", scaled_height, scaled_width);

    // Allocate CUDA Array
    returnValue = (cudaError_t)cudaMallocArray( &cu_array, &channelDesc, width ,height);
    if(returnValue != cudaSuccess){
        printf("\n got error cudaMallocArray %d", returnValue);
        return -1;
    }else{
        printf("cudaMallocArray success %d\n", returnValue);
    }
    returnValue = (cudaError_t)cudaMemcpyToArray( cu_array, 0, 0, data ,height*width*sizeof(unsigned char) , cudaMemcpyHostToDevice);

    if(returnValue != cudaSuccess){
        printf("\n Got error while running CUDA API Array Copy, %d %s\r\n", returnValue, cudaGetErrorString(returnValue));
        return -1;
    }

    // Step 1. Specifiy texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array =  cu_array;
    // Step 2. Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    // x dimension address mode
    texDesc.addressMode[0] = cudaAddressModeClamp;
    // y dimension address mode
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Step 3. Create texture object,
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    if(returnValue != cudaSuccess){
        printf("\n Got error while running CUDA API Bind Texture");
    }

    cudaMalloc(&d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char));

    dim3 dimBlock(32,32,1);
    dim3 dimGrid(scaled_width/dimBlock.x, scaled_height/dimBlock.y, 1);

    createResizedImage<<<dimGrid, dimBlock>>>(d_scaled_data, scaled_width, 1/scaling_ratio, texObj);

    returnValue = (cudaError_t)(returnValue | cudaDeviceSynchronize());

    returnValue = (cudaError_t)(returnValue | cudaMemcpy(scaled_data, d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    if(returnValue != cudaSuccess){
        printf("\n Got error while running CUDA API kernel");
    }

    // Step 5. Destroy texture object
    cudaDestroyTextureObject(texObj);

    src_write_pgm(outputStr, scaled_data, scaled_height, scaled_width, "####");
    // Storing the image with the detections,



    if(data != NULL){
        free(data);
    }
    if( scaled_data!= NULL){ 
        free(scaled_data);
    }
    if(cu_array != NULL){
        cudaFreeArray(cu_array);
    }
    if(d_scaled_data != NULL){
        cudaFree(d_scaled_data);
    }
    return 0;
}