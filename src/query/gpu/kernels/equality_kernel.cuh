//
// Created by xabdomo on 4/20/25.
//

#ifndef EQUALITY_KERNEL_CUH
#define EQUALITY_KERNEL_CUH



namespace EqualityKernel {

    __global__ void equality_kernel_int64_t(
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
    );
};



#endif //EQUALITY_KERNEL_CUH
