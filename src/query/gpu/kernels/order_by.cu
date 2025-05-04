//
// Created by xabdomo on 5/2/25.
//

#include "order_by.cuh"

__global__ void OrderBy::histogram_kernel_indexed(
    const int64_t*  data,
    const index_t* indices,
    index_t* histogram,
    index_t* pins,
    size_t num_elements,
    index_t mask_bits,
    index_t shift_bits,
    index_t num_pins
    ) {

    extern __shared__ index_t shared_hist[];

    for (index_t i = threadIdx.x; i < num_pins; i += blockDim.x) {
        shared_hist[i] = 0;
    }

    __syncthreads();

    for (
            index_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < num_elements;
            i += blockDim.x * gridDim.x) {


        index_t idx   = indices[i];
        int64_t sample = data[idx];
        index_t pin = (sample >> shift_bits) & mask_bits;

        for (int j = 0;j < num_pins;j++) {
            pins[i + num_elements * j] = 0;
        }

        pins[i + pin * num_elements] = 1;

        atomicAdd(&shared_hist[pin], (index_t) 1);
    }

    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_pins; i += blockDim.x) {
        atomicAdd(&histogram[i], (uint32_t) shared_hist[i]);
    }
}

__global__ void OrderBy::radix_scatter_pass(
        const int64_t*  data,
        const index_t* indices_in,
        index_t*       indices_out,
        index_t*       pin_offsets,        // size: num_bins, initialized from prefix_sums
        index_t*       local_offsets,
        size_t num_elements,
        index_t mask_bits,
        index_t shift_bits
    ) {

    for (
            index_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < num_elements;
            i += blockDim.x * gridDim.x) {

        index_t idx = indices_in[i];
        int64_t sample = data[idx];
        index_t pin = (sample >> shift_bits) & mask_bits;

        index_t pos = (pin == 0 ? 0 : pin_offsets[pin - 1]) + local_offsets[i + pin * num_elements] - 1;
        indices_out[pos] = idx;
    }
}

__global__ void OrderBy::efficient_local_prefix_sum(const uint32_t* input, uint32_t *output, int n, int pins, uint32_t *aux) {
    extern __shared__ uint32_t temp[]; // 2d array of size : [blockSize * 2][pins]

    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    size_t t = threadIdx.x;

    for (int i = 0;i < pins;i++) {
        temp[t * pins + i]                = 0;
        temp[(t + blockDim.x) * pins + i] = 0;

        if (idx < n              ) temp[t * pins + i]                = input[idx * pins + i];
        if (idx + blockDim.x < n ) temp[(t + blockDim.x) * pins + i] = input[(idx + blockDim.x) * pins + i];
    }

    size_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const size_t ai = factor * ( 2 * t + 1 ) - 1;
            const size_t bi = factor * ( 2 * t + 2 ) - 1;
            for (int i = 0;i < pins;i++) {
                temp[bi * pins + i] += temp[ai * pins + i];
            }
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        for (int i = 0;i < pins;i++) {
            temp[(blockDim.x * 2 - 1) * pins + i] = 0;
        }
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const size_t ai = stride * ( 2 * t + 1 ) - 1;
            const size_t bi = stride * ( 2 * t + 2 ) - 1;
            for (int i = 0;i < pins;i++) {
                const size_t val = temp[ai * pins + i];
                temp[ai * pins + i]  = temp[bi * pins + i];
                temp[bi * pins + i] += val;
            }
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) {
        for (int i = 0;i < pins;i++) {
            aux[blockIdx.x * pins + i] = temp[(blockDim.x * 2 - 1) * pins + i] + input[(blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1) * pins + i];
        }
    }

    __syncthreads();

    for (int i = 0;i < pins;i++) {
        if (idx < n             ) output[idx * pins + i]                = temp[t * pins + i]                + input[idx * pins + i];
        if (idx + blockDim.x < n) output[(idx + blockDim.x) * pins + i] = temp[(t + blockDim.x) * pins + i] + input[(idx + blockDim.x) * pins + i];
    }
}

__global__ void OrderBy::add_local_aux(uint32_t *input, int n, int pins, const uint32_t *aux) {
    size_t idx = (blockIdx.x + 1) * blockDim.x * 2 + threadIdx.x;

    if   (idx >= n) return;
    for (int i = 0;i < pins;i++) {
        input[idx * pins + i]              = aux[blockIdx.x * pins + i] + input[idx * pins + i];
    }

    if   (idx + blockDim.x >= n) return;
    for (int i = 0;i < pins;i++) {
        input[(idx + blockDim.x) * pins + i] = aux[blockIdx.x * pins + i] + input[(idx + blockDim.x) * pins + i];
    }
}
