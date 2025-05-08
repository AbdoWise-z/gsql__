//
// Created by xabdomo on 4/20/25.
//

#ifndef EQUALITY_KERNEL_CUH
#define EQUALITY_KERNEL_CUH

#include "utils/murmur_hash3_cuda.cuh"
#include "constants.hpp"
#include "inequality_kernel.cuh"
#include "tensor_kernels.cuh"

namespace EqualityKernel {

    /// ===========================
    /// Hash Accelerated Kernels
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T* col_2,
        size_t  col_1_size,
        size_t  col_2_size,

        // hash params
        const size_t* hash_table,
        size_t  hash_ext_size,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= tileShape[table_1_i]) return;

        auto col1_index = index + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto val = col_1[col1_index];
        auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(double), SEED);
        auto hash_index = hash % col_2_size;
        hash_index = hash_index * hash_ext_size;

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];

        while (hash_table[hash_index] != static_cast<size_t>(-1)) {
            auto col2_index = hash_table[hash_index];

            hash_index = hash_index + 1;
            hash_index = hash_index % (col_2_size * hash_ext_size);

            if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
                continue;
            }

            auto val2 = col_2[col2_index];
            if (InequalityKernel::cmp(val, val2) == 0) {
                pos[table_2_i] = col2_index - tileOffset[table_2_i];
                result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
            }
        }
    }


    // fixme: in-consistency with normal search, we don't hash null values, so value pairs with <null, null> will never happen in hash search ..
    /// ===========================
    /// Hash Accelerated Kernels, null support
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const char* col_1_nulls,
        const T* col_2,
        const char* col_2_nulls,
        size_t  col_1_size,
        size_t  col_2_size,

        // hash params
        const size_t* hash_table,
        size_t  hash_ext_size,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= tileShape[table_1_i]) return;

        auto col1_index = index + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto val = col_1[col1_index];
        if (col_1_nulls[col1_index]) return; // we are null
        auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(double), SEED);
        auto hash_index = hash % col_2_size;
        hash_index = hash_index * hash_ext_size;

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];

        while (hash_table[hash_index] != static_cast<size_t>(-1)) {
            auto col2_index = hash_table[hash_index];

            hash_index = hash_index + 1;
            hash_index = hash_index % (col_2_size * hash_ext_size);

            if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
                continue;
            }

            auto val2 = col_2[col2_index];
            if (col_2_nulls[col2_index]) continue; // should never happen but anyway ...
            if (InequalityKernel::cmp(val, val2) == 0) {
                pos[table_2_i] = col2_index - tileOffset[table_2_i];
                result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
            }
        }
    }


    /// ===========================
    /// Normal Kernels (2D)
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T* col_2,
        size_t  col_1_size,
        size_t  col_2_size,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t iTh = blockIdx.x * blockDim.x + threadIdx.x;
        size_t jTh = blockIdx.y * blockDim.y + threadIdx.y;

        if (iTh >= tileShape[table_1_i]) return;
        if (jTh >= tileShape[table_2_i]) return;

        auto col1_index = iTh + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto col2_index = jTh + tileOffset[table_2_i];
        if (col2_index >= col_2_size) return;

        auto val1 = col_1[col1_index];
        auto val2 = col_2[col2_index];

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];
        pos[table_2_i] = col2_index - tileOffset[table_2_i];

        if (InequalityKernel::cmp(val1, val2) == 0) {
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }


    /// ===========================
    /// Normal Kernels (2D), null support
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T* col_2,
        const char* col_1_nulls,
        const char* col_2_nulls,
        size_t  col_1_size,
        size_t  col_2_size,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t iTh = blockIdx.x * blockDim.x + threadIdx.x;
        size_t jTh = blockIdx.y * blockDim.y + threadIdx.y;

        if (iTh >= tileShape[table_1_i]) return;
        if (jTh >= tileShape[table_2_i]) return;

        auto col1_index = iTh + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto col2_index = jTh + tileOffset[table_2_i];
        if (col2_index >= col_2_size) return;

        auto val1 = col_1[col1_index];
        auto val2 = col_2[col2_index];

        if (col_1_nulls[col1_index] != col_2_nulls[col2_index]) return; // both are null or both are equal ...

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];
        pos[table_2_i] = col2_index - tileOffset[table_2_i];

        if (col_1_nulls[col1_index] || InequalityKernel::cmp(val1, val2) == 0) {
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }


    /// ===========================
    /// Normal Kernels (1D - Col , literal)
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T  literal,
        size_t  col_1_size,

        // masking params
        size_t *mask,
        size_t table_1_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t iTh = blockIdx.x * blockDim.x + threadIdx.x;

        if (iTh >= tileShape[table_1_i]) return;

        auto col1_index = iTh + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;


        auto val1 = col_1[col1_index];

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];

        if (InequalityKernel::cmp(val1, literal) == 0) {
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }


    /// ===========================
    /// Normal Kernels (1D - Col , literal), null support
    /// ===========================
    template<typename T>
    __global__ void equality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T    literal,
        const char* col_1_nulls,
        const char literal_is_null,
        size_t  col_1_size,

        // masking params
        size_t *mask,
        size_t table_1_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    ) {
        size_t iTh = blockIdx.x * blockDim.x + threadIdx.x;

        if (iTh >= tileShape[table_1_i]) return;

        auto col1_index = iTh + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto val1 = col_1[col1_index];
        if (literal_is_null != col_1_nulls[col1_index]) return;

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];

        if (literal_is_null || InequalityKernel::cmp(val1, literal) == 0) {
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }


    /// ===========================
    /// Normal Kernels (1D - Col , literal, only compare date)
    /// ===========================
    __global__ void equality_kernel_date(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const dateTime* col_1,
        const dateTime  literal,
        size_t  col_1_size,

        // masking params
        size_t *mask,
        size_t table_1_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    );


    /// ===========================
    /// Normal Kernels (1D - Col , literal, only compare time)
    /// ===========================
    __global__ void equality_kernel_time(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const dateTime* col_1,
        const dateTime  literal,
        size_t  col_1_size,

        // masking params
        size_t *mask,
        size_t table_1_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset
    );
};



#endif //EQUALITY_KERNEL_CUH
