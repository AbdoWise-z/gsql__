//
// Created by xabdomo on 4/15/25.
//

#ifndef STORE_HPP
#define STORE_HPP

#include <unordered_map>
#include <string>
#include "db/table.hpp"

extern std::unordered_map<std::string, table*> global_tables;

namespace Cfg {
    extern size_t HashTableExtendableSize;

    extern size_t maxTensorElements;

    std::vector<size_t> getTileSizeFor(const std::vector<size_t>& inputSize);
}



#endif //STORE_HPP
