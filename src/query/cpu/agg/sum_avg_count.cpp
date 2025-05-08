//
// Created by xabdomo on 4/16/25.
//

#include "sum_avg_count.hpp"

#include "query/errors.hpp"
#include "db/table.hpp"

tval Agg::CPU::sum(column *col) {
    auto size = col->data.size() - col->nullsCount;
    if (size == 0) {
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

    tval _sum = ValuesHelper::copy(col->data[0], col->type);
    for (size_t i = 1;i < col->data.size(); ++i) {
        auto item =  col->data[i];
        auto nil = col->nulls[i];
        if (nil) continue;

        auto old = _sum;
        _sum = ValuesHelper::add(old, item, col->type);
        ValuesHelper::deleteValue(old, col->type);
    }

    auto ret = ValuesHelper::copy(_sum, col->type);
    ValuesHelper::deleteValue(_sum, col->type);
    return ret;
}

tval Agg::CPU::avg(column *col) {
    auto _sum = sum(col);
    auto size = col->data.size() - col->nullsCount;

    switch (col->type) {
        case STRING:
            throw UnsupportedOperationError("Cannot aggregate (avg) over a string column");
        case INTEGER:
            _sum.d = static_cast<double>(_sum.i) / size;
            break;
        case FLOAT:
            _sum.d = ((double) _sum.d) / size;
            break;
        case DateTime:
            _sum.t->day    = _sum.t->day      / (size);
            _sum.t->month  = _sum.t->month    / (size);
            _sum.t->year   = _sum.t->year     / (size);
            _sum.t->hour   = _sum.t->hour     / (size);
            _sum.t->minute = _sum.t->minute   / (size);
            _sum.t->second = _sum.t->second   / (size);
            break;
    }
    return _sum;
}

tval Agg::CPU::count(const column *col) {
    return ValuesHelper::create_from(static_cast<int64_t>(col->data.size() - col->nullsCount));
}

tval Agg::CPU::count(const table *t) {
    return ValuesHelper::create_from(static_cast<int64_t>(t->size()));
}
