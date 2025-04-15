//
// Created by xabdomo on 4/15/25.
//

#ifndef FROM_RESOLVER_HPP
#define FROM_RESOLVER_HPP
#include <hsql/sql/Table.h>

#include "db/table.hpp"


namespace FromResolver {
    std::unordered_map<std::string, table*> resolve(hsql::TableRef*);
};



#endif //FROM_RESOLVER_HPP
