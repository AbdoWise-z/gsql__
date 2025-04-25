//
// Created by xabdomo on 4/25/25.
//

#ifndef RESOLVE_RESULT_HPP
#define RESOLVE_RESULT_HPP

#include "../db/table.hpp"

namespace FromResolver {
    struct ResolveResult {
        std::vector<std::set<std::string>> table_names;
        std::vector<table*> tables;
        std::vector<bool> isTemporary;
    };
}

#endif //RESOLVE_RESULT_HPP
