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
        auto hash = MurmurHash3_x64_64(&data[i], sizeof(tval), 0);
        hash = hash % data.size();

        while (hashed[hash].size() == ex_size) {     // this bucket is filled
            hash = (hash + 1) % data.size();
        }

        hashed[hash].push_back(i);
    }
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
