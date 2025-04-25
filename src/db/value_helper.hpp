//
// Created by xabdomo on 4/13/25.
//

#ifndef VALUE_HELPER_HPP
#define VALUE_HELPER_HPP

#include <cstdint>
#include <csv.hpp>
#include <string>

#include "typing.hpp"
#include "utils/murmur_hash3.hpp"

#include "constants.hpp"


struct dateTime {
    ushort year;
    ushort month;
    ushort day;

    u_char hour;
    u_char minute;
    u_char second;
};

union tval {
    std::string* s;
    int64_t      i;
    double       d;
    dateTime*    t;
};

int cmp(const tval& a, const tval& b, const DataType& t);

size_t sizeOf(const tval& v, const DataType& t);

tval copy(tval v, const DataType& t);

inline uint64_t hash(const tval& v, const DataType& t) {
    return MurmurHash3_x64_64((t == STRING || t == DateTime) ? static_cast<const void*>(v.s) : static_cast<const void*>(&v), sizeOf(v, t), SEED);
}

inline tval create_from(const std::string& str) {
    tval res;
    res.s = new std::string(str);
    return res;
}

inline tval create_from(int64_t i) {
    tval res;
    res.i = i;
    return res;
}


inline tval create_from(double d) {
    tval res;
    res.d = d;
    return res;
}

void deleteValue(tval, DataType);

std::string to_string(tval, DataType);


#endif //VALUE_HELPER_HPP
