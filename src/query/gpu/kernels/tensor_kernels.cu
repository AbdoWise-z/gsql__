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

__device__ __host__ void TensorKernel::unmap(size_t *shape, size_t *pos, size_t index, size_t size) {
    size_t remaining = index;
    for (int i = 0; i < size; ++i) {
        pos[i] = remaining % shape[i];
        remaining /= shape[i];
    }
}

__device__ __host__ size_t TensorKernel::map(size_t *indices, size_t *shape, size_t size) {
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

__global__ void TensorKernel::logical_and(char *a, char *b, size_t size, char *out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] && b[idx];
    }
}

__global__ void TensorKernel::logical_or(char *a, char *b, size_t size, char *out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] || b[idx];
    }
}

__global__ void TensorKernel::logical_not(char *a, size_t size, char *out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = !a[idx];
    }
}


__global__ void TensorKernel::sparce_tensor_iterator(const char *a, size_t *out, size_t size) {
    __shared__ int index;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) index = 0; // init the shared memory

    if (idx >= size) return;

    __syncthreads();

    auto val = a[idx];
    if (val == 1) out[atomicAdd(&index, 1) + blockIdx.x * blockDim.x] = idx;

    __syncthreads();

    if (threadIdx.x == 0 && index != blockDim.x) out[atomicAdd(&index, 1) + blockIdx.x * blockDim.x] = -1;
}

__global__ void TensorKernel::efficient_prefix_sum(size_t* input, size_t* output, int n, size_t* aux) {
    extern __shared__ size_t temp[];

    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    size_t t = threadIdx.x;

    temp[t]              = 0;
    temp[t + blockDim.x] = 0;

    if (idx < n)              temp[t]              = input[idx];
    if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

    size_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const size_t ai = factor * ( 2 * t + 1 ) - 1;
            const size_t bi = factor * ( 2 * t + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        temp[blockDim.x * 2 - 1] = 0;
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const size_t ai = stride * ( 2 * t + 1 ) - 1;
            const size_t bi = stride * ( 2 * t + 2 ) - 1;
            const size_t val = temp[ai];

            temp[ai]  = temp[bi];
            temp[bi] += val;
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) aux[blockIdx.x] = temp[blockDim.x * 2 - 1] + input[blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1];

    __syncthreads();

    if (idx < n)              output[idx]              = temp[t] + input[idx];
    if (idx + blockDim.x < n) output[idx + blockDim.x] = temp[t + blockDim.x] + input[idx + blockDim.x];
}

__global__ void TensorKernel::efficient_prefix_sum(char* input, size_t* output, int n, size_t* aux) {
    extern __shared__ size_t temp[];

    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    size_t t = threadIdx.x;

    temp[t]              = 0;
    temp[t + blockDim.x] = 0;

    if (idx < n)              temp[t]              = input[idx];
    if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

    size_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const size_t ai = factor * ( 2 * t + 1 ) - 1;
            const size_t bi = factor * ( 2 * t + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        temp[blockDim.x * 2 - 1] = 0;
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const size_t ai = stride * ( 2 * t + 1 ) - 1;
            const size_t bi = stride * ( 2 * t + 2 ) - 1;
            const size_t val = temp[ai];

            temp[ai]  = temp[bi];
            temp[bi] += val;
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) aux[blockIdx.x] = temp[blockDim.x * 2 - 1] + input[blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1];

    __syncthreads();

    if (idx < n)              output[idx]              = temp[t] + input[idx];
    if (idx + blockDim.x < n) output[idx + blockDim.x] = temp[t + blockDim.x] + input[idx + blockDim.x];
}

__global__ void TensorKernel::add_aux(size_t* input, int n, const size_t* aux) {
    size_t idx = (blockIdx.x + 1) * blockDim.x * 2 + threadIdx.x;

    if   (idx >= n) return;
    input[idx]              = aux[blockIdx.x] + input[idx];

    if   (idx + blockDim.x >= n) return;
    input[idx + blockDim.x] = aux[blockIdx.x] + input[idx + blockDim.x];
}

__global__ void TensorKernel::efficient_prefix_sum_index_t(index_t *input, index_t *output, int n, index_t *aux) {
    extern __shared__ index_t temp_[];

    index_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    index_t t = threadIdx.x;

    temp_[t]              = 0;
    temp_[t + blockDim.x] = 0;

    if (idx < n)              temp_[t]              = input[idx];
    if (idx + blockDim.x < n) temp_[t + blockDim.x] = input[idx + blockDim.x];

    index_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const index_t ai = factor * ( 2 * t + 1 ) - 1;
            const index_t bi = factor * ( 2 * t + 2 ) - 1;
            temp_[bi] += temp_[ai];
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        temp_[blockDim.x * 2 - 1] = 0;
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const index_t ai = stride * ( 2 * t + 1 ) - 1;
            const index_t bi = stride * ( 2 * t + 2 ) - 1;
            const index_t val = temp_[ai];

            temp_[ai]  = temp_[bi];
            temp_[bi] += val;
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) aux[blockIdx.x] = temp_[blockDim.x * 2 - 1] + input[blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1];

    __syncthreads();

    if (idx < n)              output[idx]              = temp_[t] + input[idx];
    if (idx + blockDim.x < n) output[idx + blockDim.x] = temp_[t + blockDim.x] + input[idx + blockDim.x];
}

__global__ void TensorKernel::add_aux(index_t *input, int n, const index_t *aux) {
    index_t idx = (blockIdx.x + 1) * blockDim.x * 2 + threadIdx.x;

    if   (idx >= n) return;
    input[idx]              = aux[blockIdx.x] + input[idx];

    if   (idx + blockDim.x >= n) return;
    input[idx + blockDim.x] = aux[blockIdx.x] + input[idx + blockDim.x];
}


