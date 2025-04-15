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

    void buildSortedIndexes ();

    // fixme: maybe find a better way to handle collisions ?
    void buildHashedIndexes (int ex_size);

    bool isSortIndexed() const;
    bool isHashIndexed() const;

    ~column();
};



#endif //COLUMN_HPP
