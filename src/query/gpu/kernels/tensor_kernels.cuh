//
// Created by xabdomo on 4/20/25.
//

#ifndef TENSOR_KERNALS_CUH
#define TENSOR_KERNALS_CUH

#include <cstdint>

#include "order_by.cuh"


namespace TensorKernel {
    __global__ void fill_kernel(char *output_data, char value, size_t size);
    __global__ void fill_kernel(char *output_data, size_t dataSize, char value, size_t* center_pos, size_t *mask, size_t *shape, size_t maskSize);
    __device__ __host__ void unmap(size_t* mask, size_t* pos, size_t index, size_t size);
    __device__ __host__ size_t map(size_t* indices, size_t* shape, size_t size);
    __global__ void extend_plain_kernel(char *output_data, size_t dataSize, size_t *mask, size_t *shape, size_t maskSize);

    __global__ void logical_and(char* a, char* b, size_t size, char* out);
    __global__ void logical_or (char* a, char* b, size_t size, char* out);
    __global__ void logical_not(char* a, size_t size, char* out);

    __global__ void sparce_tensor_iterator(const char* a, size_t* out, size_t size);

    __global__ void efficient_prefix_sum(char* input, size_t* output, int n, size_t* aux);
    __global__ void efficient_prefix_sum(size_t* input, size_t* output, int n, size_t* aux);
    __global__ void add_aux(size_t* input, int n, const size_t* aux);

    __global__ void efficient_prefix_sum_index_t(index_t* input, index_t* output, int n, index_t* aux);
    __global__ void add_aux(index_t* input, int n, const index_t* aux);

    template<typename T>
    __global__ void fill_kernel(T* output_data, T value, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output_data[idx] = value;
        }
    }
};



#endif //TENSOR_KERNALS_CUH
