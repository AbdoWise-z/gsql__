//
// Created by xabdomo on 4/15/25.
//

#include "select_executor.hpp"

#include <iomanip>
#include <hsql/sql/SelectStatement.h>

#include "filter_applier.hpp"
#include "from_resolver.hpp"
#include "agg/max_min.hpp"
#include "agg/sum_avg_count.hpp"
#include "db/table.hpp"
#include "query/errors.hpp"
#include "utils/konsol.hpp"
#include "utils/string_utils.hpp"

#define SELECT_DEBUG

table* SelectExecutor::Execute(hsql::SQLStatement *statement) {
    const auto* stmnt = dynamic_cast<hsql::SelectStatement*>(statement);
    if (stmnt == nullptr) {
        throw std::invalid_argument("Not a Select SQL statement");
    }

    auto* from = stmnt->fromTable;
    if (!from) {
        throw std::runtime_error("idk how you made a select query without a from ...");
    }

    // resolve the input
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

    auto where = stmnt->whereClause;
    auto limit = stmnt->limit;

    // apply the Where filter
    auto intermediate = FilterApplier::apply(&query_input, where, limit);
    auto result = ConstructTable(intermediate, &query_input);
    delete intermediate;

    // clean up any temp tables created in the process
    for (int i = 0;i < query_input.tables.size();i++) {
        // clean up
        if (query_input.isTemporary[i]) {
            delete query_input.tables[i];
        }
    }

    // perform projection / aggregation
    auto final_result = new table();
    auto projection = stmnt->selectList;
    size_t tableSize = 0;
    for (const auto expr : *projection) {
        if (expr->type == hsql::kExprStar) {
            // add all columns
            for (int j = 0;j < result.result->headers.size();j++) {
                final_result->headers.push_back(result.result->headers[j]);
                final_result->columns.push_back(result.result->columns[j]->copy());
            }
        } else if (expr->type == hsql::kExprColumnRef) {
            // add the column

            std::string col_name = expr->name;
            std::string table_name;

            if (expr->table != nullptr) {
                table_name = expr->table; // limit to specific table
            }

            auto idx = getColumn(&result, table_name, col_name);
            if (idx == -1) {
                throw NoSuchColumnError(table_name + "." + col_name);
            } else if (idx == -2) {
                throw NoSuchTableError(table_name);
            }

            std::string alias = result.result->headers[idx];
            if (expr->alias) {
                alias = expr->alias;
            }

            final_result->headers.push_back(alias);
            if (tableSize == 0 || tableSize == result.result->columns[idx]->data.size()) {
                final_result->columns.push_back(result.result->columns[idx]->copy());
                tableSize = result.result->columns[idx]->data.size();
            } else {
                throw TableSizeMismatch();
            }
        } else if (expr->type == hsql::kExprFunctionRef) {
            auto params = expr->exprList;
            if (params->size() != 1) {
                throw UnsupportedOperationError("Only functions such as count, max, sum, avg are supported with one param");
            }

            auto param = (*params)[0];
            if (param->type != hsql::kExprColumnRef) {
                throw UnsupportedOperationError("Only functions such as count, max, sum, avg are supported with one param (column name)");
            }

            if (tableSize == 0 || tableSize == 1) {
                tableSize = 1;
            } else {
                throw TableSizeMismatch();
            }

            std::string col_name = param->name;
            std::string table_name;

            if (param->table != nullptr) {
                table_name = param->table; // limit to specific table
            }

            auto idx = getColumn(&result, table_name, col_name);
            if (idx == -1) {
                throw NoSuchColumnError(table_name + "." + col_name);
            } else if (idx == -2) {
                throw NoSuchTableError(table_name);
            }

            std::string alias = result.result->headers[idx];
            if (expr->alias) {
                alias = expr->alias;
            }
            final_result->headers.push_back(alias);

            auto col = result.result->columns[idx];
            // now we apply the actual function ..
            std::string func_name = expr->name;

            if (func_name == "sum") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = col->type;
                final_result->columns.back()->data.push_back(Agg::sum(col));
            } else if (func_name == "count") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = INTEGER;
                final_result->columns.back()->data.push_back(Agg::count(col));
            } else if (func_name == "avg") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = FLOAT;
                final_result->columns.back()->data.push_back(Agg::avg(col));
            } else if (func_name == "min") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = col->type;
                final_result->columns.back()->data.push_back(Agg::min(col));
            } else if (func_name == "max") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = col->type;
                final_result->columns.back()->data.push_back(Agg::max(col));
            } else {
                throw UnsupportedOperationError("Function not supported");
            }
        }
    }

    delete result.result;

    // there is no memory leak .. the IDE is tripping.
    // ReSharper disable once CppDFAMemoryLeak
    return final_result;
}

SelectExecutor::ConstructionResult SelectExecutor::ConstructTable(
    tensor<char, CPU>* intermediate,
    const FromResolver::ResolveResult* input
    ) {
    size_t resultSize = intermediate->totalSize();

    // ReSharper disable once CppDFAMemoryLeak
    auto result = new table();
    std::vector<std::unordered_set<std::string>> col_source;

    for (int t_idx = 0;t_idx < input->table_names.size();t_idx++) {
        const auto t = input->tables[t_idx];
        for (int i = 0;i < t->headers.size();i++) {
            result->headers.push_back(t->headers[i]);
            auto col = new column();
            col->type = t->columns[i]->type;
            result->columns.push_back(col);
            col_source.push_back(input->table_names[t_idx]);
        }
    }


    for (size_t i = 0; i < resultSize; i++) {
        if ((*intermediate)[i]) {
            auto tuple_index = intermediate->unmap(i);
            int j = 0;
            for (int m = 0;m < input->tables.size();m++) {
                auto name = input->table_names[m];
                const auto t = input->tables[m];
                for (int k = 0;k < t->headers.size();k++) {
                    auto val = t->columns[k]->data[tuple_index[m]];
                    result->columns[j]->data.push_back(copy(val, t->columns[k]->type));
                    j++;
                }
            }
        }
    }

    return {
        .result = result,
        .col_source = col_source
    };
}

int SelectExecutor::getColumn(const ConstructionResult *result, const std::string& table, const std::string& column) {
    bool table_found = false;
    for (int i = 0;i < result->col_source.size();i++) {
        if (table == "" || result->col_source[i].contains(table)) {
            if (result->result->headers[i] == column) {
                return i;
            }
            table_found = true;
        }
    }

    if (table_found) return -1;
    return -2; // didn't even find the table
}
