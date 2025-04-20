//
// Created by xabdomo on 4/16/25.
//

#include "max_min.hpp"

#include "query/errors.hpp"

//#define MAX_MIN_DEBUG

tval Agg::CPU::max(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (max) over a string column");

    if (col->type == INTEGER) {
        if (col->isSortIndexed()) return create_from(col->data[col->sorted.back()].i);

        int64_t _max = std::numeric_limits<int64_t>::min();
        for (auto item: col->data) {
            _max = _max > item.i ? _max : item.i;
        }

        return create_from(_max);
    }

    if (col->type == FLOAT) {
        if (col->isSortIndexed()) return create_from(col->data[col->sorted.back()].d);

        double _max = std::numeric_limits<double>::min();
        for (auto item: col->data) {
            _max = _max > item.d ? _max : item.d;
        }

        return create_from(_max);
    }

    return create_from("ERROR IN AGG MAX");
}

tval Agg::CPU::min(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (min) over a string column");

    if (col->type == INTEGER) {

        if (col->isSortIndexed()) {
#ifdef MAX_MIN_DEBUG
            std::cout << "Agg::min result=" << col->data[col->sorted[0]].i << " (index)" << std::endl;
#endif
            return create_from(col->data[col->sorted[0]].i);
        }

        int64_t _min = std::numeric_limits<int64_t>::max();
        for (auto item: col->data) {
            _min = _min < item.i ? _min : item.i;
        }
#ifdef MAX_MIN_DEBUG
        std::cout << "Agg::min result=" << _min << std::endl;
#endif
        return create_from(_min);
    }

    if (col->type == FLOAT) {
        if (col->isSortIndexed()) {
#ifdef MAX_MIN_DEBUG
            std::cout << "Agg::min result=" << col->data[col->sorted[0]].d << " (index)" << std::endl;
#endif
            return create_from(col->data[col->sorted[0]].d);
        }

        double _min = std::numeric_limits<double>::max();
        for (auto item: col->data) {
            _min = _min < item.d ? _min : item.d;
        }

#ifdef MAX_MIN_DEBUG
        std::cout << "Agg::min result=" << _min << std::endl;
#endif
        return create_from(_min);
    }

    return create_from("ERROR IN AGG MIN");
}
