//
// Created by xabdomo on 4/20/25.
//

#ifndef INEQUALITY_KERNEL_CUH
#define INEQUALITY_KERNEL_CUH

#include "db/column.hpp"

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

        column::SortedSearchType operation
    );


    template <typename T>
    __device__ int lower_bound(const T* data, const size_t* index, int n, const T& target) {
        int low = 0;
        int high = n;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (data[index[mid]] < target) {
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
            if (data[index[mid]] > target) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        if (data[low] > target) return -1;
        return low;
    }
}



#endif //INEQUALITY_KERNEL_CUH
