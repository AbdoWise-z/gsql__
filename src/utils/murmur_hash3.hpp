//
// Created by xabdomo on 4/13/25.
//

#ifndef MURMUR_HASH3_HPP
#define MURMUR_HASH3_HPP

#include <cstdint>
#include <cstring>


struct hash128 {
    uint64_t h1, h2;
};

void MurmurHash3_x64_128(const void *key, int len, uint32_t seed, hash128& out);

inline uint64_t MurmurHash3_x64_64(const void *key, int len, uint32_t seed) {
    hash128 h{};
    MurmurHash3_x64_128(key, len, seed, h);
    return h.h1 ^ h.h2;
}

#endif //MURMUR_HASH3_HPP
