//
// Created by xabdomo on 4/16/25.
//

#include "max_min.hpp"

#include "query/errors.hpp"
#include "query/gpu/gpu_function_interface.cuh"

//#define MAX_MIN_DEBUG

tval Agg::GPU::max(column *col) {
    // if (col->type == STRING)
    //     throw UnsupportedOperationError("Cannot aggregate (max) over a string column");
    //
    // if (col->type == INTEGER) {
    //     if (col->isSortIndexed()) return ValuesHelper::create_from(col->data[col->sorted.back()].i);
    //
    //     int64_t _max = std::numeric_limits<int64_t>::min();
    //     for (auto item: col->data) {
    //         _max = _max > item.i ? _max : item.i;
    //     }
    //
    //     return ValuesHelper::create_from(_max);
    // }
    //
    // if (col->type == FLOAT) {
    //     if (col->isSortIndexed()) return ValuesHelper::create_from(col->data[col->sorted.back()].d);
    //
    //     double _max = std::numeric_limits<double>::min();
    //     for (auto item: col->data) {
    //         _max = _max > item.d ? _max : item.d;
    //     }
    //
    //     return ValuesHelper::create_from(_max);
    // }

    if (col->data.size() == 0) {
        switch (col->type) {
            case STRING:
                return ValuesHelper::create_from("");
            case INTEGER:
                return ValuesHelper::create_from(static_cast<int64_t>(0));
            case FLOAT:
                return ValuesHelper::create_from(static_cast<double>(0));
            case DateTime:
                return ValuesHelper::create_from(dateTime{});
        }
    }
    return GFI::max(col);

    // return ValuesHelper::create_from("ERROR IN AGG MAX");
}

tval Agg::GPU::min(column *col) {
//     if (col->type == STRING)
//         throw UnsupportedOperationError("Cannot aggregate (min) over a string column");
//
//     if (col->type == INTEGER) {
//
//         if (col->isSortIndexed()) {
// #ifdef MAX_MIN_DEBUG
//             std::cout << "Agg::min result=" << col->data[col->sorted[0]].i << " (index)" << std::endl;
// #endif
//             return ValuesHelper::create_from(col->data[col->sorted[0]].i);
//         }
//
//         int64_t _min = std::numeric_limits<int64_t>::max();
//         for (auto item: col->data) {
//             _min = _min < item.i ? _min : item.i;
//         }
// #ifdef MAX_MIN_DEBUG
//         std::cout << "Agg::min result=" << _min << std::endl;
// #endif
//         return ValuesHelper::create_from(_min);
//     }
//
//     if (col->type == FLOAT) {
//         if (col->isSortIndexed()) {
// #ifdef MAX_MIN_DEBUG
//             std::cout << "Agg::min result=" << col->data[col->sorted[0]].d << " (index)" << std::endl;
// #endif
//             return ValuesHelper::create_from(col->data[col->sorted[0]].d);
//         }
//
//         double _min = std::numeric_limits<double>::max();
//         for (auto item: col->data) {
//             _min = _min < item.d ? _min : item.d;
//         }
//
// #ifdef MAX_MIN_DEBUG
//         std::cout << "Agg::min result=" << _min << std::endl;
// #endif
//         return ValuesHelper::create_from(_min);
//     }
//
//     return ValuesHelper::create_from("ERROR IN AGG MIN");

    if (col->data.size() == 0) {
        switch (col->type) {
            case STRING:
                return ValuesHelper::create_from("");
            case INTEGER:
                return ValuesHelper::create_from(static_cast<int64_t>(0));
            case FLOAT:
                return ValuesHelper::create_from(static_cast<double>(0));
            case DateTime:
                return ValuesHelper::create_from(dateTime{});
        }
    }

    return GFI::min(col);
}
