//
// Created by xabdomo on 4/13/25.
//

#include "value_helper.hpp"

#include <string>


int cmp(const tval& a, const tval& b, const DataType& t) {
    switch (t) {
        case STRING:
            return strcmp(a.s->c_str(), b.s->c_str());
        case INTEGER:
            return a.i > b.i ? 1 : (a.i < b.i ? -1 : 0);
        case FLOAT:
            return a.d > b.d ? 1 : (a.d < b.d ? -1 : 0);
        case DateTime:
            if (a.t->year - b.t->year) return a.t->year - b.t->year;
            if (a.t->month - b.t->month) return a.t->month - b.t->month;
            if (a.t->day - b.t->day) return a.t->day - b.t->day;
            if (a.t->hour - b.t->hour) return a.t->hour - b.t->hour;
            if (a.t->minute - b.t->minute) return a.t->minute - b.t->minute;
            return a.t->second - b.t->second;
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
        case DateTime:
            return sizeof(dateTime);
    }

    return 0;
}

tval copy(const tval v, const DataType& t) {
    tval result{};

    if (t == STRING) {
        result.s = new std::string(*v.s);
    } else if (t == INTEGER) {
        result.i = static_cast<int64_t>(v.i);
    } else if (t == FLOAT) {
        result.d = static_cast<double>(v.d);
    } else {
        result.t = new dateTime(*v.t);
    }

    return result;
}

void deleteValue(tval v, DataType t) {
    if (t == STRING) delete v.s;
    if (t == DateTime) delete v.t;

    v.s = nullptr; // v.t == nullptr;
}

std::string to_string(tval v, DataType t) {
    switch (t) {
        case STRING:
            return *v.s;
        case INTEGER:
            return std::to_string(v.i);
        case FLOAT:
            return std::to_string(v.d);
        case DateTime:
            return  std::to_string(v.t->year)   + "-" +
                    std::to_string(v.t->month)  + "-" +
                    std::to_string(v.t->day)    + " " +
                    std::to_string(v.t->hour)   + ":" +
                    std::to_string(v.t->minute) + ":" +
                    std::to_string(v.t->second);
    }

    return "__non_type__";
}
