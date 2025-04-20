//
// Created by xabdomo on 4/13/25.
//

#ifndef MURMUR_HASH3_CU_HPP
#define MURMUR_HASH3_CU_HPP

#include <cstdint>

struct hash128_cuda {
    uint64_t h1, h2;
};

__device__ __host__ void MurmurHash3_x64_128_cuda(const void *key, int len, uint32_t seed, hash128_cuda* out);

__device__ __host__ uint64_t MurmurHash3_x64_64_cuda(const void *key, int len, uint32_t seed);

#endif //MURMUR_HASH3_CU_HPP
