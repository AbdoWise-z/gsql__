//
// Created by xabdomo on 4/15/25.
//

#include "store.hpp"


std::unordered_map<std::string, table*> global_tables;

namespace Cfg {
    size_t HashTableExtendableSize = 4;
}