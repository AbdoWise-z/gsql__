//
// Created by xabdomo on 4/15/25.
//

#ifndef SELECT_EXECUTOR_HPP
#define SELECT_EXECUTOR_HPP

#include <unordered_map>
#include <hsql/sql/SQLStatement.h>

#include "from_resolver.hpp"
#include "tensor/tensor.hpp"
#include "db/table.hpp"

namespace SelectExecutor {
    struct ConstructionResult {
        table* result;
        std::vector<std::unordered_set<std::string>> col_source;
    };

    table* Execute(hsql::SQLStatement* statement);

    ConstructionResult ConstructTable(
       tensor<char, CPU>* intermediate,
       const FromResolver::ResolveResult* input
       );
};



#endif //SELECT_EXECUTOR_HPP
