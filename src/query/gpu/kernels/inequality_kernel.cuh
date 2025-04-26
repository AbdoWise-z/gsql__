//
// Created by xabdomo on 4/20/25.
//

#ifndef INEQUALITY_KERNEL_CUH
#define INEQUALITY_KERNEL_CUH

#include "tensor_kernels.cuh"
#include "db/column.hpp"
#include "db/value_helper.hpp"

namespace InequalityKernel {

    __global__ void Inequality_kernel_int64_t(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const int64_t* col_1,
        const int64_t* col_2,
        size_t  col_1_size,
        size_t  col_2_size,

        // sort params
        const size_t* sorted_index,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset,

        // actual operation
        column::SortedSearchType operation
    );

    __device__ inline int cmp(int64_t a, int64_t b) {
        return 1 * (a > b) + -1 * (a < b);
    }

    __device__ inline int cmp(const double& a, const double& b) {
        return 1 * (a > b) + -1 * (a < b);
    }

    __device__ inline int cmp(const char* a, const char* b) {
        int i = 0;
        while (a[i] != '\0' && a[i] == b[i]) {
            i++;
        }
        return 1 * (a[i] > b[i]) + -1 * (a[i] < b[i]);
    }

    __device__ inline int cmp(const dateTime& a, const dateTime& b) {
        int64_t a_v = a.year * 31104000 + a.month * 2592000 + a.day * 86000 + a.hour * 3600 + a.minute * 60 + a.second;
        int64_t b_v = b.year * 31104000 + b.month * 2592000 + b.day * 86000 + b.hour * 3600 + b.minute * 60 + b.second;
        return 1 * (a_v > b_v) + -1 * (a_v < b_v);
    }


    template <typename T>
    __device__ int lower_bound(const T* data, const size_t* index, int n, const T& target) {
        int low = 0;
        int high = n;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (cmp(data[index[mid]], target) < 0) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    template <typename T>
    __device__ int upper_bound(const T* data, const size_t* index, int n, const T& target) {
        int low = 0;
        int high = n;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (cmp(data[index[mid]], target) > 0) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        if (cmp(data[low], target) > 0) return -1;
        return low;
    }

    template<typename T>
    __global__ void inequality_kernel(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const T* col_1,
        const T* col_2,
        size_t  col_1_size,
        size_t  col_2_size,

        // sort params
        const size_t* sorted_index,

        // masking params
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,

        // tiling params
        size_t* tileShape,
        size_t* tileOffset,

        // actual operation
        column::SortedSearchType operation
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= tileShape[table_1_i]) return;

        auto col1_index = index + tileOffset[table_1_i];
        if (col1_index >= col_1_size) return;

        auto val = col_1[col1_index];

        size_t pos[MAX_TENSOR_DIMS];

        for (int i = 0;i < tablesCount; i++) {
            pos[i] = 0; // store the data in the zero-th plain
        }

        pos[table_1_i] = col1_index - tileOffset[table_1_i];

        // search logic
        if (operation == column::SST_GT) {
            auto start = lower_bound(col_2, sorted_index, col_2_size, val);
            while (start < col_2_size) {
                pos[table_2_i] = sorted_index[start] - tileOffset[table_2_i];
                result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
                start ++;
            }
        } else { // must be less than
            auto start = upper_bound(col_2, sorted_index, col_2_size, val);
            while (start >= 0) {
                pos[table_2_i] = sorted_index[start] - tileOffset[table_2_i];
                result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
                start --;
            }
        }
    }

    template<typename T>
    __global__ void inequality_kernel(
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
        size_t* tileOffset,

        // actual operation
        column::SortedSearchType operation
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

        auto cmp_value = InequalityKernel::cmp(val1, val2);

        if (cmp_value > 0 && operation == column::SST_GT || cmp_value < 0 && operation == column::SST_LT) {
            pos[table_2_i] = col2_index - tileOffset[table_2_i];
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }
}



#endif //INEQUALITY_KERNEL_CUH
