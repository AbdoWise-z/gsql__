//
// Created by xabdomo on 4/20/25.
//

#include "tensor_kernels.cuh"
#include "constants.hpp"
#include "utils/murmur_hash3_cuda.cuh"

__global__ void TensorKernel::fill_kernel(char *output_data, char value, size_t size)  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output_data[idx] = value;
    }
}

__global__ void TensorKernel::fill_kernel(char *output_data, size_t dataSize, char value, size_t* center_pos, size_t *mask, size_t *shape, size_t maskSize) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dataSize) {
        return;
    }

    size_t pos[MAX_TENSOR_DIMS];
    unmap(shape, pos, idx, maskSize);

    bool _set = true;
    for (int i = 0; i < maskSize; ++i) {
        if (!mask[i] && pos[i] != center_pos[i]) {
            _set = false;
            break;
        }
    }

    if (_set)
        output_data[idx] = value;
}

__device__ void TensorKernel::unmap(size_t *shape, size_t *pos, size_t index, size_t size) {
    size_t remaining = index;
    for (int i = 0; i < size; ++i) {
        pos[i] = remaining % shape[i];
        remaining /= shape[i];
    }
}

size_t TensorKernel::map(size_t *indices, size_t *shape, size_t size) {
    size_t index = 0;
    size_t acc = 1;
    for (size_t i = 0;i < size;i++) {
        index += indices[i] * acc;
        acc *= shape[i];
    }
    return index;
}

__global__ void TensorKernel::extend_plain_kernel(
        char *output_data,
        size_t dataSize,
        size_t *mask,
        size_t *shape,
        size_t maskSize
    ) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataSize) {
        return;
    }

    size_t pos[MAX_TENSOR_DIMS];
    unmap(shape, pos, idx, maskSize);

    for (int i = 0; i < maskSize; ++i) {
        if (mask[i]) {
            pos[i] = 0; // load the data from the zero-th plain
            break;
        }
    }

    output_data[idx] = output_data[map(pos, shape, maskSize)];
}
