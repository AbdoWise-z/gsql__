//
// Created by xabdomo on 4/13/25.
//

#ifndef VALUE_HELPER_HPP
#define VALUE_HELPER_HPP

#include <cstdint>
#include <csv.hpp>

#include "typing.hpp"
#include "utils/murmur_hash3.hpp"

#define SEED 2147483647  // a prime number from the boi Euler :)    M31

union tval {
    const char* s;
    int64_t     i;
    double      d;
};

int cmp(const tval& a, const tval& b, const DataType& t);

size_t sizeOf(const tval& v, const DataType& t);

inline uint64_t hash(const tval& v, const DataType& t) {
    return MurmurHash3_x64_64(&v, sizeOf(v, t), 0);
}

#endif //VALUE_HELPER_HPP
