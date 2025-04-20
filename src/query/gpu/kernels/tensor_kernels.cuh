//
// Created by xabdomo on 4/20/25.
//

#ifndef TENSOR_KERNALS_CUH
#define TENSOR_KERNALS_CUH



namespace TensorKernel {
    __global__ void fill_kernel(char *output_data, char value, size_t size);
    __global__ void fill_kernel(char *output_data, size_t dataSize, char value, size_t* center_pos, size_t *mask, size_t *shape, size_t maskSize);
    __device__ __host__ void unmap(size_t* mask, size_t* pos, size_t index, size_t size);
    __device__ __host__ size_t map(size_t* indices, size_t* shape, size_t size);
    __global__ void extend_plain_kernel(char *output_data, size_t dataSize, size_t *mask, size_t *shape, size_t maskSize);

    __global__ void logical_and(char* a, char* b, size_t size, char* out);
    __global__ void logical_or (char* a, char* b, size_t size, char* out);
    __global__ void logical_not(char* a, size_t size, char* out);
};



#endif //TENSOR_KERNALS_CUH
