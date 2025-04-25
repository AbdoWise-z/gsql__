//
// Created by xabdomo on 4/15/25.
//

#include <iostream>
#include <hsql/sql/SelectStatement.h>


#include "gpu_executor.hpp"
#include "errors.hpp"
#include "store.hpp"
#include "gpu/select_executor.hpp"
#include "db/table.hpp"

static table* statementExecutor(hsql::SQLStatement* statement, TableMap& tables) {
    switch (statement->type()) {
        case hsql::kStmtSelect:
            return SelectExecutor::GPU::Execute(statement, tables);
        default:
            throw UnsupportedOperationError(statement->type());
    }
}

std::vector<table*> GPUExecutor::executeQuery(const hsql::SQLParserResult &query, TableMap& tables) {
    if (query.isValid() == false) {
        throw std::invalid_argument("Invalid SQL query");
    }

    std::vector<table*> results;
    for (const auto s: query.getStatements()) {
        results.push_back(statementExecutor(s, tables));
    }

    return results;
}
