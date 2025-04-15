//
// Created by xabdomo on 4/15/25.
//

#include "select_executor.hpp"

#include <iomanip>
#include <hsql/sql/SelectStatement.h>

#include "filter_applier.hpp"
#include "from_resolver.hpp"
#include "db/table.hpp"
#include "utils/konsol.hpp"

#define SEELCT_DEBUG

table* SelectExecutor::Execute(hsql::SQLStatement *statement) {
    auto* stmnt = dynamic_cast<hsql::SelectStatement*>(statement);
    if (stmnt == nullptr) {
        throw std::invalid_argument("Not a Select SQL statement");
    }

    auto* from = stmnt->fromTable;
    if (!from) {
        throw std::runtime_error("idk how you made a select query without a from ...");
    }

    auto query_input = FromResolver::resolve(from);

#ifdef SEELCT_DEBUG
    std::cout << "Query IN:" << std::endl;
    for (const auto& [k, v]: query_input) {
        std::cout << "T\t" << std::setw(24) << std::left << color(k, CYAN_FG) << "at " << std::hex << MAGENTA_FG << v << RESET_FG << std::endl;
    }
#endif

    auto expr = stmnt->whereClause;
    auto limit = stmnt->limit;

    auto intermediate = FilterApplier::apply(query_input, expr, limit);

#ifdef SELECT_DEBUG

#endif
    return nullptr;
}
