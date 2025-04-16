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
#include "utils/string_utils.hpp"

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

#ifdef SELECT_DEBUG
    std::cout << "Query IN:" << std::endl;
    for (int i = 0;i < query_input.table_names.size();i++) {
        auto k = StringUtils::join(query_input.table_names[i].begin(), query_input.table_names[i].end());
        k = StringUtils::limit(k, 24 * 2);
        auto v = query_input.tables[i];
        auto t = query_input.isTemporary[i];
        std::cout << "T\t" << std::setw(24) << std::left << color(k, CYAN_FG) << "at " << std::hex << MAGENTA_FG << v << RESET_FG << "\t" << (t ? "(r)" : "") << std::endl;
    }
#endif

    auto expr = stmnt->whereClause;
    auto limit = stmnt->limit;

    auto intermediate = FilterApplier::apply(&query_input, expr, limit);
    auto result = ConstructTable(intermediate, &query_input);
    delete intermediate;

#ifdef SELECT_DEBUG
#endif
    return result.result;
}

SelectExecutor::ConstructionResult SelectExecutor::ConstructTable(
    tensor<char, CPU>* intermediate,
    const FromResolver::ResolveResult* input
    ) {
    size_t resultSize = intermediate->totalSize();

    const auto result = new table();
    for (const auto& t: input->tables) {
        for (int i = 0;i < t->headers.size();i++) {
            result->headers.push_back(t->headers[i]);
            column col;
            col.type = t->columns[i].type;
            result->columns.push_back(col);
        }
    }

    std::vector<std::unordered_set<std::string>> col_source;

    for (size_t i = 0; i < resultSize; i++) {
        if ((*intermediate)[i]) {
            auto tuple_index = intermediate->unmap(i);
            int j = 0;
            for (int m = 0;m < input->tables.size();m++) {
                auto name = input->table_names[m];
                const auto t = input->tables[m];
                for (int k = 0;k < t->headers.size();k++) {
                    auto val = t->columns[k].data[tuple_index[m]];
                    result->columns[j].data.push_back(copy(val, t->columns[k].type));
                    j++;
                    col_source.push_back(name);
                }
            }
        }
    }

    return {
        .result = result,
        .col_source = col_source
    };
}
