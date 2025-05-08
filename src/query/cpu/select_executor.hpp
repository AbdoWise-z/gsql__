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

namespace SelectExecutor::CPU {
    struct ConstructionResult {
        table* result;
        std::vector<std::set<std::string>> col_source;
    };

    std::pair<std::set<std::string>, table*> Execute(hsql::SQLStatement* statement, TableMap& tables, FromResolver::ResolveResult inject = {});

    ConstructionResult ConstructTable(
       tensor<char, Device::CPU>* intermediate,
       const FromResolver::ResolveResult* input
       );

    void AppendTable(
       tensor<char, Device::CPU>* intermediate,
       const FromResolver::ResolveResult* input,
       const std::vector<size_t> &offset,
       const table* result
       );

    int getColumn(const ConstructionResult* result, const std::string& table, const std::string& column);

    typedef std::vector<size_t> MultDimVector;
    std::vector<MultDimVector> Schedule(const std::vector<size_t> &inputSize);
};



#endif //SELECT_EXECUTOR_HPP
