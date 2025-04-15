//
// Created by xabdomo on 4/13/25.
//

#include "column.hpp"

#include "value_helper.hpp"
#include "utils/murmur_hash3.hpp"

void column::buildSortedIndexes() {
    sorted.clear();

    for (int i = 0;i < data.size(); i++) {
        sorted.push_back(i);
    }

    std::ranges::sort(sorted, [this](const auto& a, const auto& b) {
        return cmp(data[a], data[b], type);
    });
}

void column::buildHashedIndexes(int ex_size) {
    hashed.clear();

    for (int i = 0;i < data.size(); i++) { // at most as big as the data size will be
        hashed.emplace_back();
    }

    for (int i = 0;i < data.size(); i++) {
        auto hash_v = hash(data[i], type);
        hash_v = hash_v % data.size();

        while (hashed[hash_v].size() == ex_size) {     // this bucket is filled
            hash_v = (hash_v + 1) % data.size();
        }

        hashed[hash_v].push_back(i);
    }
}

std::vector<size_t> column::hashSearch(const tval v) const {
    auto hash_v = hash(v, type);
    hash_v = hash_v % data.size();
    std::vector<size_t> result;
    while (hashed[hash_v].size() > 0) {
        for (auto item: hashed[hash_v]) {
            if (cmp(data[item], v, type) == 0) {
                result.push_back(item);
            }
        }
        hash_v = (hash_v + 1) % data.size();
    }

    return result;
}

bool column::isSortIndexed() const {
    return sorted.size() == data.size();
}

bool column::isHashIndexed() const {
    return hashed.size() == data.size();
}

column::~column() {
    if (type == STRING) {
        for (const auto i: data) {
            delete i.s;
        }
    }
}
