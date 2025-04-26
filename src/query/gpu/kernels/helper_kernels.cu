//
// Created by xabdomo on 4/27/25.
//

#include "helper_kernels.cuh"

__global__ void HelperKernels::strlen(const char *c, size_t *result) {
    size_t size = 0;
    while (c[size] != '\0') size++;
    result[0] = size;
}
