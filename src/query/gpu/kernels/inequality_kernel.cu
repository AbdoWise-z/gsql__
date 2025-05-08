//
// Created by xabdomo on 4/20/25.
//

#include "inequality_kernel.cuh"

#include "constants.hpp"
#include "tensor_kernels.cuh"


__global__ void InequalityKernel::Inequality_kernel_int64_t(
        char *result,
        size_t dataSize,
        size_t tablesCount,
        const int64_t *col_1,
        const int64_t *col_2,
        size_t col_1_size,
        size_t col_2_size,
        const size_t *sorted_index,
        size_t *mask,
        size_t table_1_i,
        size_t table_2_i,
        size_t *tileShape,
        size_t *tileOffset,
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

extern __device__ const char* EMPTY_CHAR = "";
