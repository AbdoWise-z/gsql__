//
// Created by xabdomo on 4/20/25.
//

#include "equality_kernel.cuh"

#include "utils/murmur_hash3_cuda.cuh"
#include "constants.hpp"
#include "tensor_kernels.cuh"


__global__ void EqualityKernel::equality_kernel_int64_t(
        // result params
        char* result,
        size_t dataSize,
        size_t tablesCount,

        // data params
        const int64_t* col_1,
        const int64_t* col_2,
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
    auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(int64_t), SEED);
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
        if (val == val2) {
            pos[table_2_i] = col2_index - tileOffset[table_2_i];
            result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
        }
    }
}
