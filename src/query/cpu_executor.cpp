//
// Created by xabdomo on 4/15/25.
//

#include <iostream>
#include <hsql/sql/SelectStatement.h>


#include "cpu_executor.hpp"
#include "errors.hpp"
#include "cpu/select_executor.hpp"
#include "db/table.hpp"

static table* statementExecutor(hsql::SQLStatement* statement, TableMap& tables) {
    switch (statement->type()) {
        case hsql::kStmtSelect:
            return SelectExecutor::CPU::Execute(statement, tables).second;
        default:
            throw UnsupportedOperationError(statement->type());
    }
}

std::vector<table*> CPUExecutor::executeQuery(const hsql::SQLParserResult &query, TableMap& tables) {
    if (query.isValid() == false) {
        throw std::invalid_argument("Invalid SQL query");
    }

    std::vector<table*> results;
    for (const auto s: query.getStatements()) {
        results.push_back(statementExecutor(s, tables));
    }

    return results;
}
