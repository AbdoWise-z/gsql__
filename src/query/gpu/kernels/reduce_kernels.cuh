//
// Created by xabdomo on 4/27/25.
//

#ifndef REDUCE_KERNELS_CUH
#define REDUCE_KERNELS_CUH

#include "inequality_kernel.cuh"
#include "store.hpp"


namespace ReduceKernel {

    __device__ inline int64_t  add(int64_t  a, int64_t   b) { return a + b; }
    __device__ inline double   add(double   a, double    b) { return a + b; }
    __device__ inline dateTime add(dateTime a, dateTime  b) {
        dateTime res;
        res.day = a.day + b.day;
        res.month = a.month + b.month;
        res.year = a.year + b.year;
        res.hour = a.hour + b.hour;
        res.minute = a.minute + b.minute;
        res.second = a.second + b.second;
        return res;
    }



    template <typename T>
    __global__ void max(
        T* input,
        size_t n,
        T* blockResult
    ) {
        __shared__ T temp[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = input[idx % n];
        temp[t + blockDim.x] = input[(idx + blockDim.x) % n];

        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                if (InequalityKernel::cmp(temp[bi], temp[ai]) < 0) {
                    temp[bi] = temp[ai];
                }
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
    }


    template <typename T>
    __global__ void min(
        T* input,
        size_t n,
        T* blockResult
    ) {
        __shared__ T temp[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = input[idx % n];
        temp[t + blockDim.x] = input[(idx + blockDim.x) % n];

        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                if (InequalityKernel::cmp(temp[bi], temp[ai]) > 0) {
                    temp[bi] = temp[ai];
                }
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
    }


    template <typename T>
    __global__ void sum(
        T* input,
        size_t n,
        T* blockResult
    ) {
        __shared__ T temp[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = {0};
        temp[t + blockDim.x] = {0};

        if (idx < n)              temp[t]              = input[idx];
        if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                temp[bi] = add(temp[bi], temp[ai]);
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
    }

};



#endif //REDUCE_KERNELS_CUH
