//
// Created by xabdomo on 4/16/25.
//

#include "sum_avg_count.hpp"

#include "query/errors.hpp"
#include "db/table.hpp"

tval Agg::CPU::sum(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (sum) over a string column");

    if (col->type == INTEGER) {
        int64_t sum = 0;
        for (auto item: col->data) {
            sum += item.i;
        }

        return ValuesHelper::create_from(sum);
    }

    if (col->type == FLOAT) {
        double sum = 0.0;
        for (auto item: col->data) {
            sum += item.d;
        }

        return ValuesHelper::create_from(sum);
    }

    return ValuesHelper::create_from("ERROR IN AGG SUM");
}

tval Agg::CPU::avg(column *col) {
    if (col->type == STRING)
        throw UnsupportedOperationError("Cannot aggregate (avg) over a string column");

    if (col->data.empty()) {
        return ValuesHelper::create_from("Inf");
    }

    if (col->type == INTEGER) {
        double sum = 0;
        for (auto item: col->data) {
            sum += item.i;
        }

        return ValuesHelper::create_from(sum / col->data.size());
    }

    if (col->type == FLOAT) {
        double sum = 0.0;
        for (auto item: col->data) {
            sum += item.d;
        }

        return ValuesHelper::create_from(sum / col->data.size());
    }

    return ValuesHelper::create_from("ERROR IN AGG SUM");
}

tval Agg::CPU::count(const column *col) {
    return ValuesHelper::create_from(static_cast<int64_t>(col->data.size()));
}

tval Agg::CPU::count(const table *t) {
    return ValuesHelper::create_from(static_cast<int64_t>(t->size()));
}
