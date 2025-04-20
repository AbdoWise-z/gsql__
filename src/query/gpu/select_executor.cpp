//
// Created by xabdomo on 4/15/25.
//

#include "select_executor.hpp"

#include <iomanip>
#include <hsql/sql/SelectStatement.h>

#include "filter_applier.hpp"
#include "from_resolver.hpp"
#include "store.hpp"
#include "agg/max_min.hpp"
#include "agg/sum_avg_count.hpp"
#include "db/table.hpp"
#include "query/errors.hpp"
#include "utils/konsol.hpp"
#include "utils/string_utils.hpp"

#define SELECT_DEBUG

table* SelectExecutor::GPU::Execute(hsql::SQLStatement *statement) {

    const auto* stmnt = dynamic_cast<hsql::SelectStatement*>(statement);
    if (stmnt == nullptr) {
        throw std::invalid_argument("Not a Select SQL statement");
    }

    auto* from = stmnt->fromTable;
    if (!from) {
        throw std::runtime_error("idk how you made a select query without a from ...");
    }

    // resolve the input
    auto query_input = FromResolver::GPU::resolve(from);

#ifdef SELECT_DEBUG
    std::cout << "Query: " << std::endl;
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

    std::vector<size_t> inputSize;
    for (auto k : query_input.tables) {
        if (k->columns.size() > 0) {
            inputSize.push_back(k->size());
        } else {
            inputSize.push_back(0);
        }
    }

    auto tileSize = Cfg::getTileSizeFor(inputSize);
    // apply the Where filter
    ConstructionResult result {
        .result = nullptr,
        .col_source = {}
    };

    auto tiles = Schedule(inputSize);

#ifdef SELECT_DEBUG
    std::cout << "Input size: [ ";
    for (auto i : inputSize) {
        std::cout << std::dec << i << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Tile size: [ ";
    for (auto i : tileSize) {
        std::cout << std::dec << i << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Total number of tiles: " << std::dec << tiles.size() << std::endl;
#endif

    size_t debug_current_tile = 0;
#ifdef SELECT_DEBUG
    std::cout << "Progress: 0/" << std::dec << tiles.size();
    std::cout.flush();
#endif
    for (auto tile: tiles) {
        auto intermediate = FilterApplier::GPU::apply(
            &query_input,
            where,
            limit,
            tile,
            tileSize
            );

        auto cpu_version = intermediate->toCPU();
        if (result.result == nullptr) {
            result = ConstructTable(&cpu_version, tileSize, &query_input);
        } else {
            AppendTable(&cpu_version, tileSize, &query_input, tile, result.result);
        }


#ifdef SELECT_DEBUG
        std::cout << "\rProgress: " << std::dec << (++debug_current_tile) << "/" << std::dec << tiles.size();
        std::cout.flush();
#endif
        delete intermediate;
    }

#ifdef SELECT_DEBUG
    std::cout << std::endl;
#endif

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
                final_result->columns.back()->data.push_back(Agg::GPU::sum(col));
            } else if (func_name == "count") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = INTEGER;
                final_result->columns.back()->data.push_back(Agg::GPU::count(col));
            } else if (func_name == "avg") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = FLOAT;
                final_result->columns.back()->data.push_back(Agg::GPU::avg(col));
            } else if (func_name == "min") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = col->type;
                final_result->columns.back()->data.push_back(Agg::GPU::min(col));
            } else if (func_name == "max") {
                final_result->columns.push_back(new column());
                final_result->columns.back()->type = col->type;
                final_result->columns.back()->data.push_back(Agg::GPU::max(col));
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

SelectExecutor::GPU::ConstructionResult SelectExecutor::GPU::ConstructTable(
    tensor<char, Device::CPU>* intermediate,
    const std::vector<size_t>& tileSize,
    const FromResolver::GPU::ResolveResult* input
    ) {
    size_t resultSize = std::accumulate(tileSize.begin(), tileSize.end(), 1, std::multiplies<size_t>());

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

    auto iter_size = intermediate->totalSize();
    for (size_t i = 0; i < resultSize; i++) {
        if ((*intermediate)[i % iter_size]) {
            auto tuple_index = ::unmap(tileSize, i);

            bool _in_bound = true;
            for (int m = 0;m < input->table_names.size();m++) {
                if (input->tables[m]->size() <= tuple_index[m]) {
                    _in_bound = false;
                    break;
                }
            }
            if (!_in_bound) continue;

            int j = 0;
            for (int m = 0;m < input->tables.size();m++) {
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

void SelectExecutor::GPU::AppendTable(
    tensor<char, Device::CPU> *intermediate,
    const std::vector<size_t>& tileSize,
    const FromResolver::GPU::ResolveResult *input,
    const std::vector<size_t> &offset,
    const table *result
    ) {

    std::vector<size_t> pos(offset.size(), 0);

    size_t resultSize = std::accumulate(tileSize.begin(), tileSize.end(), 1, std::multiplies<size_t>());
    auto iter_size = intermediate->totalSize();

    for (size_t i = 0; i < resultSize; i++) {
        if ((*intermediate)[i % iter_size]) {

            auto tuple_index = ::unmap(tileSize, i);

            for (int j = 0;j < pos.size();j++) {
                pos[j] = offset[j] + tuple_index[j];
            }

            bool _in_bound = true;
            for (int m = 0;m < input->table_names.size();m++) {
                if (input->tables[m]->size() <= pos[m]) {
                    _in_bound = false;
                    break;
                }
            }
            if (!_in_bound) continue;

            int j = 0;
            for (int m = 0;m < input->tables.size();m++) {
                const auto t = input->tables[m];
                for (int k = 0;k < t->headers.size();k++) {
                    auto val = t->columns[k]->data[pos[m]];
                    result->columns[j]->data.push_back(copy(val, t->columns[k]->type));
                    j++;
                }
            }
        }
    }
}

int SelectExecutor::GPU::getColumn(const ConstructionResult *result, const std::string& table, const std::string& column) {
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

static void ScheduleInternalRecursive(
        std::vector<SelectExecutor::GPU::MultDimVector>& result,
        const std::vector<size_t>& inputSize,
        const std::vector<size_t>& tileSize,
        SelectExecutor::GPU::MultDimVector pos,
        int dim = 0
    ) {

    if (dim != inputSize.size()) {
        size_t iter = (inputSize[dim] - 1) / tileSize[dim] + 1;
        for (int i = 0;i < iter;i++) {
            pos[dim] = i * tileSize[dim];
            ScheduleInternalRecursive(
                result,
                inputSize,
                tileSize,
                pos,
                dim + 1);
        }
    } else {
        result.push_back(pos);
    }
}

std::vector<SelectExecutor::GPU::MultDimVector> SelectExecutor::GPU::Schedule(const std::vector<size_t> &inputSize) {
    std::vector<MultDimVector> result;
    std::vector<size_t> _startPos(inputSize.size(), 0);
    std::vector<size_t> _tileSize = Cfg::getTileSizeFor(inputSize);
    ScheduleInternalRecursive(result, inputSize, _tileSize, _startPos, 0);
    return result;
}
