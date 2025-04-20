//
// Created by xabdomo on 4/13/25.
//

#ifndef COLUMN_HPP
#define COLUMN_HPP

#include <vector>
#include "typing.hpp"

union tval;

class column {
public:
    std::vector<tval> data;
    DataType type;

    std::vector<size_t>              sorted;
    std::vector<std::vector<size_t>> hashed;
    size_t hashExSize;

    void buildSortedIndexes ();
    void buildHashedIndexes (int ex_size);

    [[nodiscard]] std::vector<size_t> hashSearch(tval) const;

    enum SortedSearchType {
        SST_GT,
        SST_LT,
        SST_GTE,
        SST_LTE
    };

    [[nodiscard]] std::vector<size_t> sortSearch(tval, SortedSearchType) const;

    [[nodiscard]] bool isSortIndexed() const;
    [[nodiscard]] bool isHashIndexed() const;

    [[nodiscard]] column* copy() const;
    ~column();
};



#endif //COLUMN_HPP
