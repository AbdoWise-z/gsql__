//
// Created by xabdomo on 4/16/25.
//

#include "sum_avg_count.hpp"

#include "query/errors.hpp"

tval Agg::GPU::sum(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (sum) over a string column");

    if (col->type == INTEGER) {
        int64_t sum = 0;
        for (auto item: col->data) {
            sum += item.i;
        }

        return create_from(sum);
    }

    if (col->type == FLOAT) {
        double sum = 0.0;
        for (auto item: col->data) {
            sum += item.d;
        }

        return create_from(sum);
    }

    return create_from("ERROR IN AGG SUM");
}

tval Agg::GPU::avg(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (avg) over a string column");

    if (col->data.empty()) {
        return create_from("Inf");
    }

    if (col->type == INTEGER) {
        double sum = 0;
        for (auto item: col->data) {
            sum += item.i;
        }

        return create_from(sum / col->data.size());
    }

    if (col->type == FLOAT) {
        double sum = 0.0;
        for (auto item: col->data) {
            sum += item.d;
        }

        return create_from(sum / col->data.size());
    }

    return create_from("ERROR IN AGG SUM");
}

tval Agg::GPU::count(const column *col) {
    return create_from(static_cast<int64_t>(col->data.size()));
}
