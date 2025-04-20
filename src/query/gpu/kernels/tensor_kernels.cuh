//
// Created by xabdomo on 4/20/25.
//

#ifndef TENSOR_KERNALS_CUH
#define TENSOR_KERNALS_CUH



namespace TensorKernel {

    // fills an entire tensor with some value
    // each thread does only one cell
    // assumes linear blocks on X-axis
    __global__ void fill_kernel(char *output_data, char value, size_t size);
    __global__ void fill_kernel(char *output_data, size_t dataSize, char value, size_t* center_pos, size_t *mask, size_t *shape, size_t maskSize);
    __device__ void unmap(size_t* mask, size_t* pos, size_t index, size_t size);
};



#endif //TENSOR_KERNALS_CUH
