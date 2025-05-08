//
// Created by xabdomo on 4/16/25.
//

#include "max_min.hpp"

#include "query/errors.hpp"

//#define MAX_MIN_DEBUG

tval Agg::CPU::max(column *col) {
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

    if (col->isSortIndexed()) {
        for (size_t i = 0;i < col->sorted.size(); ++i) {
            if (!col->nulls[col->sorted[i]]) {
                return ValuesHelper::copy(col->data[col->sorted[i]], col->type);
            }
        }

        throw std::runtime_error("__should__never__happen__");
    } else {
        tval _max = col->data[0];
        for (size_t i = 1;i < col->data.size(); ++i) {
            auto item = col->data[i];
            auto nil = col->nulls[i];
            if (nil) continue;

            if (ValuesHelper::cmp(_max, item, col->type) < 0) {
                _max = item;
            }
        }

        return ValuesHelper::copy(_max, col->type);
    }
}

tval Agg::CPU::min(column *col) {
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

    if (col->isSortIndexed()) {
        for (int i = col->sorted.size() - 1;i >= 0; --i) {
            if (!col->nulls[col->sorted[i]]) {
                return ValuesHelper::copy(col->data[col->sorted[i]], col->type);
            }
        }

        throw std::runtime_error("__should__never__happen__");
    } else {
        tval _min = col->data[0];
        for (size_t i = 1;i < col->data.size(); ++i) {
            auto item = col->data[i];
            auto nil = col->nulls[i];
            if (nil) continue;

            if (ValuesHelper::cmp(_min, item, col->type) > 0) {
                _min = item;
            }
        }

        return ValuesHelper::copy(_min, col->type);
    }
}
