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

#define SELECT_DEBUG

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
    std::vector<std::string> names_mapping;
    for (auto& [k, v]: query_input) {
        names_mapping.push_back(k);
    }

#ifdef SELECT_DEBUG
    std::cout << "Query IN:" << std::endl;
    for (const auto& [k, v]: query_input) {
        std::cout << "T\t" << std::setw(24) << std::left << color(k, CYAN_FG) << "at " << std::hex << MAGENTA_FG << v << RESET_FG << std::endl;
    }
#endif

    auto expr = stmnt->whereClause;
    auto limit = stmnt->limit;

    auto intermediate = FilterApplier::apply(query_input, expr, limit, names_mapping);
    size_t resultSize = intermediate->totalSize();

    const auto result = new table();
    for (const auto& name: names_mapping) {
        const auto t = query_input[name];

        for (int i = 0;i < t->headers.size();i++) {
            result->headers.push_back(t->headers[i]);
            column col;
            col.type = t->columns[i].type;
            result->columns.push_back(col);
        }
    }

    for (size_t i = 0; i < resultSize; i++) {
        auto match = (*intermediate)[i];
        if (match) {
            auto tuple_index = intermediate->unmap(i);

            int j = 0;
            for (int m = 0;m < names_mapping.size();m++) {
                auto name = names_mapping[m];
                const auto t = query_input[name];
                for (int k = 0;k < t->headers.size();k++) {
                    auto val = t->columns[k].data[tuple_index[m]];
                    result->columns[j].data.push_back(copy(val, t->columns[k].type));
                    j++;
                }
            }
        }
    }

    delete intermediate;

#ifdef SELECT_DEBUG
#endif
    return result;
}
