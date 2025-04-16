//
// Created by xabdomo on 4/15/25.
//

#ifndef FROM_RESOLVER_HPP
#define FROM_RESOLVER_HPP

#include <hsql/sql/Table.h>

#include "db/table.hpp"


namespace FromResolver {
    struct ResolveResult {
        std::vector<std::unordered_set<std::string>> table_names;
        std::vector<table*> tables;
        std::vector<bool> isTemporary;
    };

    ResolveResult merge(ResolveResult* a, ResolveResult* b);
    int find(ResolveResult* a, std::string tname);

    ResolveResult resolve(hsql::TableRef*);
};



#endif //FROM_RESOLVER_HPP
