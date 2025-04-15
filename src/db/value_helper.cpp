//
// Created by xabdomo on 4/13/25.
//

#include "value_helper.hpp"

#include <c++/12/cstring>


int cmp(const tval& a, const tval& b, const DataType& t) {
    switch (t) {
        case STRING:
            return strcmp(a.s->c_str(), b.s->c_str());
        case INTEGER:
            return a.i > b.i ? 1 : (a.i < b.i ? -1 : 0);
        case FLOAT:
            return a.d > b.d ? 1 : (a.d < b.d ? -1 : 0);
    }

    return 0;
}

size_t sizeOf(const tval &v, const DataType &t) {
    switch (t) {
        case STRING:
            return v.s->length();
        case INTEGER:
            return sizeof(int64_t);
        case FLOAT:
            return sizeof(double);
    }

    return 0;
}

tval copy(const tval &v, const DataType& t) {
    tval result;

    switch (t) {
        case STRING:
            result.s = new std::string(*(v.s));
        case INTEGER:
            result.i = static_cast<int64_t>(v.i);
        case FLOAT:
            result.d = static_cast<double>(v.d);
    }

    return result;
}

void deleteValue(tval v, DataType t) {
    if (t == STRING) delete v.s;
    v.s = nullptr;
}
