//
// Created by xabdomo on 4/27/25.
//

#ifndef REDUCE_KERNELS_CUH
#define REDUCE_KERNELS_CUH

#include "inequality_kernel.cuh"
#include "store.hpp"


namespace ReduceKernel {

    __device__ inline int64_t MIN_VALUE(int64_t) {
        return INT64_MIN;
    }

    __device__ inline dateTime MIN_VALUE(dateTime) {
        static const dateTime m_dateTime = {
            0,0,0,0,0,0
        };
        return m_dateTime;
    }

    __device__ inline double MIN_VALUE(double) {
        return -DBL_MAX;
    }

    __device__ inline const char* MIN_VALUE(const char*) {
        static const char* _val = "\0";
        return _val;
    }


    __device__ inline int64_t MAX_VALUE(int64_t) {
        return INT64_MAX;
    }

    __device__ inline double MAX_VALUE(double) {
        return DBL_MAX;
    }

    __device__ inline const char* MAX_VALUE(const char*) {
        static const char* _val = "\255";
        return _val;
    }

    __device__ inline dateTime MAX_VALUE(dateTime) {
        static const dateTime m_dateTime = {
            UINT16_MAX,UINT16_MAX,UINT16_MAX,255,255,255
        };
        return m_dateTime;
    }


    __device__ inline int64_t ZERO_VALUE(int64_t) {
        return 0;
    }

    __device__ inline double ZERO_VALUE(double) {
        return 0.0;
    }

    __device__ inline const char* ZERO_VALUE(const char*) {
        static const char* _val = "";
        return _val;
    }

    __device__ inline dateTime ZERO_VALUE(dateTime) {
        static const dateTime m_dateTime = {
            0,0,0,0,0,0
        };
        return m_dateTime;
    }

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
    __global__ void max_nulls(
        T* input,
        const char* nulls,
        size_t n,
        T* blockResult,
        char* blockNull
    ) {
        __shared__ T    temp[MAX_BLOCK_SIZE * 2];
        __shared__ char temp_nulls[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = input[idx % n];
        temp[t + blockDim.x] = input[(idx + blockDim.x) % n];

        temp_nulls[t]                     = nulls[idx % n];
        temp_nulls[t + blockDim.x]        = nulls[(idx + blockDim.x) % n];


        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                auto v_a = temp[ai];
                auto v_b = temp[bi];
                auto nul_a = temp_nulls[ai];
                auto nul_b = temp_nulls[bi];

                if (nul_a) v_a = MIN_VALUE(v_a);
                if (nul_b) v_b = MIN_VALUE(v_b);

                if (InequalityKernel::cmp(v_b, v_a) < 0) {
                    temp[bi]       = v_a;
                    temp_nulls[bi] = nul_a;
                }
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
        if (t == 0 && blockNull   != nullptr) blockNull[blockIdx.x]   = temp_nulls[blockDim.x * 2 - 1];
    }


    // __global__ void max_nulls_dt(
    //     dateTime* input,
    //     char* nulls,
    //     size_t n,
    //     dateTime* blockResult,
    //     char* blockNull
    // ) {
    //     __shared__ dateTime    temp[MAX_BLOCK_SIZE * 2];
    //     __shared__ char temp_nulls[MAX_BLOCK_SIZE * 2];
    //
    //     size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    //
    //     size_t t = threadIdx.x;
    //
    //     temp[t]              = input[idx % n];
    //     temp[t + blockDim.x] = input[(idx + blockDim.x) % n];
    //
    //     temp_nulls[t]                     = nulls[idx % n];
    //     temp_nulls[t + blockDim.x]        = nulls[(idx + blockDim.x) % n];
    //
    //
    //     size_t factor = 1;
    //     for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
    //         __syncthreads();
    //         if (t < stride) {
    //             const size_t ai = factor * ( 2 * t + 1 ) - 1;
    //             const size_t bi = factor * ( 2 * t + 2 ) - 1;
    //             auto v_a = temp[ai];
    //             auto v_b = temp[bi];
    //             auto nul_a = temp_nulls[ai];
    //             auto nul_b = temp_nulls[bi];
    //
    //             if (nul_a) v_a = MIN_VALUE(v_a);
    //             if (nul_b) v_b = MIN_VALUE(v_b);
    //
    //             printf("a -> nil = %d; val = %d, %d, %d, %d, %d, %d\n", nul_a, v_a.year, v_a.month, v_a.day, v_a.hour, v_a.minute, v_a.second);
    //             printf("b -> nil = %d; val = %d, %d, %d, %d, %d, %d\n", nul_b, v_b.year, v_b.month, v_b.day, v_b.hour, v_b.minute, v_b.second);
    //             printf("cmp -> %d\n", InequalityKernel::cmp(v_b, v_a));
    //
    //             if (InequalityKernel::cmp(v_b, v_a) < 0) {
    //                 temp[bi]       = v_a;
    //                 temp_nulls[bi] = nul_a;
    //             }
    //         }
    //         factor <<= 1;
    //     }
    //
    //     __syncthreads();
    //     if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
    //     if (t == 0 && blockNull   != nullptr) blockNull[blockIdx.x]   = temp_nulls[blockDim.x * 2 - 1];
    // }


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
    __global__ void min_nulls(
        T* input,
        const char* nulls,
        size_t n,
        T* blockResult,
        char* blockNull
    ) {
        __shared__ T    temp[MAX_BLOCK_SIZE * 2];
        __shared__ char temp_nulls[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = input[idx % n];
        temp[t + blockDim.x] = input[(idx + blockDim.x) % n];

        temp_nulls[t]                     = nulls[idx % n];
        temp_nulls[t + blockDim.x]        = nulls[(idx + blockDim.x) % n];


        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                auto v_a = temp[ai];
                auto v_b = temp[bi];
                auto nul_a = temp_nulls[ai];
                auto nul_b = temp_nulls[bi];

                if (nul_a) v_a = MAX_VALUE(v_a);
                if (nul_b) v_b = MAX_VALUE(v_b);

                if (InequalityKernel::cmp(v_b, v_a) > 0) {
                    temp[bi]       = v_a;
                    temp_nulls[bi] = nul_a;
                }
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
        if (t == 0 && blockNull   != nullptr) blockNull[blockIdx.x]   = temp_nulls[blockDim.x * 2 - 1];
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


    template <typename T>
    __global__ void sum_nulls(
        T* input,
        const char* nulls,
        size_t n,
        T* blockResult,
        char* blockNull
    ) {
        __shared__ T    temp[MAX_BLOCK_SIZE * 2];
        __shared__ char temp_nulls[MAX_BLOCK_SIZE * 2];

        size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        size_t t = threadIdx.x;

        temp[t]              = { 0 };
        temp[t + blockDim.x] = { 0 };

        if (idx < n)              temp[t]              = input[idx];
        if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

        temp_nulls[t]                     = nulls[idx % n];
        temp_nulls[t + blockDim.x]        = nulls[(idx + blockDim.x) % n];

        size_t factor = 1;
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (t < stride) {
                const size_t ai = factor * ( 2 * t + 1 ) - 1;
                const size_t bi = factor * ( 2 * t + 2 ) - 1;
                auto v_a = temp[ai];
                auto v_b = temp[bi];
                auto nul_a = temp_nulls[ai];
                auto nul_b = temp_nulls[bi];

                if (nul_a) v_a = ZERO_VALUE(v_a);
                if (nul_b) v_b = ZERO_VALUE(v_b);

                temp[bi] = add(v_a, v_b);

                temp_nulls[bi] = nul_a && nul_b;
            }
            factor <<= 1;
        }

        __syncthreads();
        if (t == 0 && blockResult != nullptr) blockResult[blockIdx.x] = temp[blockDim.x * 2 - 1];
        if (t == 0 && blockNull   != nullptr) blockNull[blockIdx.x]   = temp_nulls[blockDim.x * 2 - 1];
    }

};



#endif //REDUCE_KERNELS_CUH
