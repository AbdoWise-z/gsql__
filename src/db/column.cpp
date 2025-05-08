//
// Created by xabdomo on 4/13/25.
//

#include "column.hpp"

#include "value_helper.hpp"

void column::buildSortedIndexes() {
    sorted.clear();

    for (int i = 0;i < data.size(); i++) {
        sorted.push_back(i);
    }

    std::sort(sorted.begin(), sorted.end(), [this](const auto& a, const auto& b) {
        return ValuesHelper::cmp(data[a], data[b], type) < 0; // Descending ordering
    });
}

void column::buildHashedIndexes(int ex_size) {
    hashed.clear();

    hashExSize = ex_size;

    for (int i = 0;i < data.size(); i++) { // at most as big as the data size will be
        hashed.emplace_back();
    }

    for (int i = 0;i < data.size(); i++) {
        if (nulls.size() > i && nulls[i]) continue;    // don't hash null values (sort is ok tho ..)
        auto hash_v = ValuesHelper::hash(data[i], type);
        hash_v = hash_v % data.size();

        while (hashed[hash_v].size() == ex_size) {     // this bucket is filled
            hash_v = (hash_v + 1) % data.size();
        }

        hashed[hash_v].push_back(i);
    }
}

std::vector<size_t> column::hashSearch(const tval v) const {
    auto hash_v = ValuesHelper::hash(v, type);
    hash_v = hash_v % data.size();
    std::vector<size_t> result;
    while (hashed[hash_v].size() > 0) {
        for (auto item: hashed[hash_v]) {
            if (ValuesHelper::cmp(data[item], v, type) == 0) {
                result.push_back(item);
            }
        }
        hash_v = (hash_v + 1) % data.size();
    }

    return result;
}

std::vector<size_t> column::nullsSearch() const {
    std::vector<size_t> result;
    for (size_t i = 0;i < nulls.size(); i++) {
        if (nulls[i]) result.push_back(i);
    }

    return result;
}

std::vector<size_t> column::sortSearch(tval v, SortedSearchType t) const {
    std::vector<size_t> result;
    auto it = sorted.begin();
    auto it2 = sorted.begin();
    switch (t) {
        case SST_GT:
            it = std::lower_bound(sorted.begin(), sorted.end(), v, [&] (auto a, auto val) {
                return ValuesHelper::cmp(val, data[a], type) >= 0;
            });
            while (it != sorted.end()) {
                result.push_back(*it);
                ++it;
            }
            break;
        case SST_GTE:
            it = std::lower_bound(sorted.begin(), sorted.end(), v, [&] (auto a, auto val) {
                return ValuesHelper::cmp(val, data[a], type) > 0;
            });
            while (it != sorted.end()) {
                result.push_back(*it);
                ++it;
            }
            break;

        case SST_LT:
            it = std::upper_bound(sorted.begin(), sorted.end(), v, [&] (auto val, auto a) {
                return ValuesHelper::cmp(data[a], val, type) >= 0;
            });

            while (it2 != it) {
                result.push_back(*it2);
                ++it2;
            }
            break;

        case SST_LTE:
            it = std::upper_bound(sorted.begin(), sorted.end(), v, [&] (auto val, auto a) {
                return ValuesHelper::cmp(data[a], val, type) > 0;
            });

            while (it2 != it) {
                result.push_back(*it2);
                ++it2;
            }
            break;
    }

    std::vector<size_t> ret;
    for (auto& i : result) {
        if (!nulls[i]) {
            ret.push_back(i);
        }
    }

    return ret;
}

bool column::isSortIndexed() const {
    return sorted.size() == data.size() && !sorted.empty();
}

bool column::isHashIndexed() const {
    return hashed.size() == data.size() && !hashed.empty();
}

column* column::copy() const {
    auto result = new column();
    result->type = type;
    result->data.resize(data.size());
    result->nulls.resize(data.size());

    for (auto i = 0;i < data.size();i++) {
        result->data[i] = ValuesHelper::copy(data[i], type);
        result->nulls[i] = nulls[i];
    }

    result->nullsCount = nullsCount;

    return result;
}

namespace GFI {
    void clearCache(void* col);
}



column::~column() {
    for (const auto i: data) {
        ValuesHelper::deleteValue(i, type);
    }

    GFI::clearCache(this); // clear any column cache in the gpu
}
