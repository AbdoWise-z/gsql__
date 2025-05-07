//
// Created by xabdomo on 4/20/25.
//

#include "equality_kernel.cuh"

#include "utils/murmur_hash3_cuda.cuh"
#include "constants.hpp"
#include "inequality_kernel.cuh"
#include "tensor_kernels.cuh"

//
// __global__ void EqualityKernel::equality_kernel_int64_t(
//         // result params
//         char* result,
//         size_t dataSize,
//         size_t tablesCount,
//
//         // data params
//         const int64_t* col_1,
//         const int64_t* col_2,
//         size_t  col_1_size,
//         size_t  col_2_size,
//
//         // hash params
//         const size_t* hash_table,
//         size_t  hash_ext_size,
//
//         // masking params
//         size_t *mask,
//         size_t table_1_i,
//         size_t table_2_i,
//
//         // tiling params
//         size_t* tileShape,
//         size_t* tileOffset
//     ) {
//
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (index >= tileShape[table_1_i]) return;
//
//     auto col1_index = index + tileOffset[table_1_i];
//     if (col1_index >= col_1_size) return;
//
//     auto val = col_1[col1_index];
//     auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(int64_t), SEED);
//     auto hash_index = hash % col_2_size;
//     hash_index = hash_index * hash_ext_size;
//
//     size_t pos[MAX_TENSOR_DIMS];
//
//     for (int i = 0;i < tablesCount; i++) {
//         pos[i] = 0; // store the data in the zero-th plain
//     }
//
//     pos[table_1_i] = col1_index - tileOffset[table_1_i];
//
//     while (hash_table[hash_index] != static_cast<size_t>(-1)) {
//         auto col2_index = hash_table[hash_index];
//
//         hash_index = hash_index + 1;
//         hash_index = hash_index % (col_2_size * hash_ext_size);
//
//         if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
//             continue;
//         }
//
//         auto val2 = col_2[col2_index];
//         if (val == val2) {
//             pos[table_2_i] = col2_index - tileOffset[table_2_i];
//             result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
//         }
//     }
// }
//
// __global__ void EqualityKernel::equality_kernel_float_t(char *result, size_t dataSize, size_t tablesCount, const double *col_1,
//     const double *col_2, size_t col_1_size, size_t col_2_size, const size_t *hash_table, size_t hash_ext_size,
//     size_t *mask, size_t table_1_i, size_t table_2_i, size_t *tileShape, size_t *tileOffset) {
//
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (index >= tileShape[table_1_i]) return;
//
//     auto col1_index = index + tileOffset[table_1_i];
//     if (col1_index >= col_1_size) return;
//
//     auto val = col_1[col1_index];
//     auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(double), SEED);
//     auto hash_index = hash % col_2_size;
//     hash_index = hash_index * hash_ext_size;
//
//     size_t pos[MAX_TENSOR_DIMS];
//
//     for (int i = 0;i < tablesCount; i++) {
//         pos[i] = 0; // store the data in the zero-th plain
//     }
//
//     pos[table_1_i] = col1_index - tileOffset[table_1_i];
//
//     while (hash_table[hash_index] != static_cast<size_t>(-1)) {
//         auto col2_index = hash_table[hash_index];
//
//         hash_index = hash_index + 1;
//         hash_index = hash_index % (col_2_size * hash_ext_size);
//
//         if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
//             continue;
//         }
//
//         auto val2 = col_2[col2_index];
//         if (InequalityKernel::cmp(val, val2) == 0) {
//             pos[table_2_i] = col2_index - tileOffset[table_2_i];
//             result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
//         }
//     }
// }
//
// __global__ void EqualityKernel::equality_kernel_str_t(char *result, size_t dataSize, size_t tablesCount, const char **col_1,
//     const char **col_2, size_t col_1_size, size_t col_2_size, const size_t *hash_table, size_t hash_ext_size,
//     size_t *mask, size_t table_1_i, size_t table_2_i, size_t *tileShape, size_t *tileOffset) {
//
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (index >= tileShape[table_1_i]) return;
//
//     auto col1_index = index + tileOffset[table_1_i];
//     if (col1_index >= col_1_size) return;
//
//     auto val = col_1[col1_index];
//     auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(double), SEED);
//     auto hash_index = hash % col_2_size;
//     hash_index = hash_index * hash_ext_size;
//
//     size_t pos[MAX_TENSOR_DIMS];
//
//     for (int i = 0;i < tablesCount; i++) {
//         pos[i] = 0; // store the data in the zero-th plain
//     }
//
//     pos[table_1_i] = col1_index - tileOffset[table_1_i];
//
//     while (hash_table[hash_index] != static_cast<size_t>(-1)) {
//         auto col2_index = hash_table[hash_index];
//
//         hash_index = hash_index + 1;
//         hash_index = hash_index % (col_2_size * hash_ext_size);
//
//         if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
//             continue;
//         }
//
//         auto val2 = col_2[col2_index];
//         if (InequalityKernel::cmp(val, val2) == 0) {
//             pos[table_2_i] = col2_index - tileOffset[table_2_i];
//             result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
//         }
//     }
// }
//
// __global__ void EqualityKernel::equality_kernel_dt_t(char *result, size_t dataSize, size_t tablesCount, const dateTime *col_1,
//     const dateTime *col_2, size_t col_1_size, size_t col_2_size, const size_t *hash_table, size_t hash_ext_size,
//     size_t *mask, size_t table_1_i, size_t table_2_i, size_t *tileShape, size_t *tileOffset) {
//
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (index >= tileShape[table_1_i]) return;
//
//     auto col1_index = index + tileOffset[table_1_i];
//     if (col1_index >= col_1_size) return;
//
//     auto val = col_1[col1_index];
//     auto hash = MurmurHash3_x64_64_cuda(&val, sizeof(double), SEED);
//     auto hash_index = hash % col_2_size;
//     hash_index = hash_index * hash_ext_size;
//
//     size_t pos[MAX_TENSOR_DIMS];
//
//     for (int i = 0;i < tablesCount; i++) {
//         pos[i] = 0; // store the data in the zero-th plain
//     }
//
//     pos[table_1_i] = col1_index - tileOffset[table_1_i];
//
//     while (hash_table[hash_index] != static_cast<size_t>(-1)) {
//         auto col2_index = hash_table[hash_index];
//
//         hash_index = hash_index + 1;
//         hash_index = hash_index % (col_2_size * hash_ext_size);
//
//         if (col2_index < tileOffset[table_2_i] || col2_index >= tileOffset[table_2_i] + tileShape[table_2_i]) {
//             continue;
//         }
//
//         auto val2 = col_2[col2_index];
//         if (InequalityKernel::cmp(val, val2) == 0) {
//             pos[table_2_i] = col2_index - tileOffset[table_2_i];
//             result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
//         }
//     }
// }

// __global__ void EqualityKernel::equality_kernel_int64_t(char *result, size_t dataSize, size_t tablesCount, const int64_t *col_1,
//     const int64_t *col_2, size_t col_1_size, size_t col_2_size, size_t *mask, size_t table_1_i, size_t table_2_i,
//     size_t *tileShape, size_t *tileOffset) {
//
//     size_t iTh = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t jTh = blockIdx.y * blockDim.y + threadIdx.y;
//
//     if (iTh >= tileShape[table_1_i]) return;
//     if (jTh >= tileShape[table_2_i]) return;
//
//     auto col1_index = iTh + tileOffset[table_1_i];
//     if (col1_index >= col_1_size) return;
//
//     auto col2_index = jTh + tileOffset[table_2_i];
//     if (col2_index >= col_2_size) return;
//
//     auto val1 = col_1[col1_index];
//     auto val2 = col_2[col2_index];
//
//     size_t pos[MAX_TENSOR_DIMS];
//
//     for (int i = 0;i < tablesCount; i++) {
//         pos[i] = 0; // store the data in the zero-th plain
//     }
//
//     pos[table_1_i] = col1_index - tileOffset[table_1_i];
//     pos[table_2_i] = col2_index - tileOffset[table_2_i];
//
//     if (InequalityKernel::cmp(val1, val2) == 0) {
//         pos[table_2_i] = col2_index - tileOffset[table_2_i];
//         result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
//     }
// }

__global__ void EqualityKernel::equality_kernel_date(char *result, size_t dataSize, size_t tablesCount, const dateTime *col_1,
    const dateTime literal, size_t col_1_size, size_t *mask, size_t table_1_i, size_t *tileShape, size_t *tileOffset) {

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


    if (val1.day == literal.day && val1.year == literal.year && val1.month == literal.month) {
        result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
    }
}


__global__ void EqualityKernel::equality_kernel_time(char *result, size_t dataSize, size_t tablesCount, const dateTime *col_1,
    const dateTime literal, size_t col_1_size, size_t *mask, size_t table_1_i, size_t *tileShape, size_t *tileOffset) {

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


    if (val1.hour == literal.hour && val1.minute == literal.minute && val1.second == literal.second) {
        result[TensorKernel::map(pos, tileShape, tablesCount)] = 1;
    }
}
