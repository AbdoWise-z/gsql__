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
#include "query/query_optimizer.hpp"
#include "utils/konsol.hpp"
#include "utils/string_utils.hpp"

#define SELECT_DEBUG

namespace SelectExecutor::CPU {
    TableMap global_input;
}

std::pair<std::set<std::string>, table*> SelectExecutor::CPU::Execute(hsql::SQLStatement *statement, TableMap& tables, FromResolver::ResolveResult inject) {

    const auto* stmnt = dynamic_cast<hsql::SelectStatement*>(statement);
    if (stmnt == nullptr) {
        throw std::invalid_argument("Not a Select SQL statement");
    }

    auto* from = stmnt->fromTable;
    if (!from) {
        throw std::runtime_error("idk how you made a select query without a from ...");
    }

    global_input = tables;

    // resolve the input
    auto global_query_input = FromResolver::CPU::resolve(from, tables);
    global_query_input = FromResolver::merge(&global_query_input, &inject);

#ifdef SELECT_DEBUG
    std::cout << "Query: " << std::endl;
    for (int i = 0;i < global_query_input.table_names.size();i++) {
        auto k = StringUtils::join(global_query_input.table_names[i].begin(), global_query_input.table_names[i].end());
        k = StringUtils::limit(k, 42 * 2);
        auto v = global_query_input.tables[i];
        auto t = global_query_input.isTemporary[i];
        std::cout << "T\t" << std::setw(42) << std::left << color(k, CYAN_FG) << "at " << std::hex << MAGENTA_FG << v << RESET_FG << "\t" << (t ? "(r)" : "") << std::endl;
    }
#endif

    auto global_where = stmnt->whereClause;
    auto limit = stmnt->limit;

    auto sub_queries = QueryOptimizer::ReducePlan(QueryOptimizer::GeneratePlan(global_query_input, stmnt));

#ifdef SELECT_DEBUG
    std::cout << "Execution Plan:" << std::endl;
    for (int i = 0; i < sub_queries.size(); ++i) {
        auto step = sub_queries[i];
        std::cout << "\t" << color( std::to_string(i), BLUE_FG) << " -> { ";
        int j = 0;
        for (const auto& name: step.output_names) {
            std::cout << color(name, YELLOW_FG);
            if (j++ != step.output_names.size() - 1) std::cout << ", ";
        }

        std::cout << " } where " << fcolor(QueryOptimizer::exprToString(step.query), MAGENTA_FG) << ";" << std::endl;
    }
#endif

    ConstructionResult result {
        .result = nullptr,
        .col_source = {}
    };

    std::set<std::string> result_tables{};

    for (auto& step: sub_queries) {
        ConstructionResult local_result {
            .result = nullptr,
            .col_source = {}
        };

        auto query_input = step.input;
        auto where = step.query;

        bool injected = false;

        for (int i = 0; i < query_input.tables.size(); ++i) {
            if (QueryOptimizer::intersects(query_input.table_names[i], result_tables)) {
                query_input.tables[i] = result.result;
                query_input.table_names[i] = QueryOptimizer::Union(result_tables, query_input.table_names[i]);
                step.output_names.insert(result_tables.begin(), result_tables.end());
                injected = true;
            }
        }

        if (!injected && result.result != nullptr) {
            query_input.tables.push_back(result.result);
            query_input.table_names.push_back(result_tables);
            step.output_names.insert(result_tables.begin(), result_tables.end());
        }

        if (where == nullptr && query_input.tables.size() == 1) {
            // no where clause, just pass the table
            local_result.result = query_input.tables[0];
            local_result.col_source = {};
            for (const auto& h: local_result.result->headers) {
                local_result.col_source.push_back(query_input.table_names[0]);
            }

            auto prev_ptr = result.result;
            if (shouldDeleteIntermediate(global_query_input, prev_ptr)) delete prev_ptr;

            result = local_result;
            result_tables.insert(query_input.table_names[0].begin(), query_input.table_names[0].end());
            continue;
        }

        std::vector<size_t> inputSize;
        bool empty_input = false;
        for (auto k : query_input.tables) {
            empty_input = empty_input || (k->size() == 0);
            inputSize.push_back(k->size());
        }

        if (!empty_input) {
            auto tileSize = Cfg::getTileSizeFor(inputSize);
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
            for (const auto& tile: tiles) {
                auto intermediate = FilterApplier::CPU::apply(
                        &query_input,
                        where,
                        limit,
                        tile,
                        tileSize
                    );

                if (local_result.result == nullptr) {
                    local_result = ConstructTable(intermediate, &query_input);
                } else {
                    AppendTable(intermediate, &query_input, tile, local_result.result);
                }

#ifdef SELECT_DEBUG
                std::cout << "\rProgress: " << std::dec << (++debug_current_tile) << "/" << std::dec << tiles.size() << "\t";
                std::cout.flush();
#endif

                delete intermediate;
            }

#ifdef SELECT_DEBUG
            std::cout << std::endl;
#endif
        } else {
            // the input itself is empty ... so what we do is just create a new table
            auto empty_result = new table();
            std::vector<std::set<std::string>> col_source;

            for (int t_idx = 0;t_idx < query_input.table_names.size();t_idx++) {
                const auto t = query_input.tables[t_idx];
                for (int i = 0;i < t->headers.size();i++) {
                    empty_result->headers.push_back(t->headers[i]);
                    auto col = new column();
                    col->type = t->columns[i]->type;
                    empty_result->columns.push_back(col);
                    col_source.push_back(query_input.table_names[t_idx]);
                }
            }

            local_result.result = empty_result;
            local_result.col_source = col_source;
        }

        auto prev_ptr = result.result;
        if (shouldDeleteIntermediate(global_query_input, prev_ptr)) delete prev_ptr;
        result = local_result;
        result_tables = step.output_names;
    }

    // perform projection / aggregation
    ConstructionResult final_result_construction {
        .result = new table(),
        .col_source = {},
    };


    auto projection = stmnt->selectList;
    size_t tableSize = 0;
    for (const auto expr : *projection) {
        if (expr->type == hsql::kExprStar) {
            // add all columns
            for (int j = 0;j < result.result->headers.size();j++) {
                final_result_construction.result->headers.push_back(result.result->headers[j]);
                final_result_construction.col_source.push_back(result.col_source[j]);
                final_result_construction.result->columns.push_back(result.result->columns[j]->copy());
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

            final_result_construction.result->headers.push_back(alias);
            final_result_construction.col_source.push_back(result.col_source[idx]);
            if (tableSize == 0 || tableSize == result.result->columns[idx]->data.size()) {
                final_result_construction.result->columns.push_back(result.result->columns[idx]->copy());
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
            if (param->type != hsql::kExprColumnRef && param->type != hsql::kExprStar) {
                throw UnsupportedOperationError("Only functions such as count, max, sum, avg are supported with one param (column name)");
            }

            if (tableSize == 0 || tableSize == 1) {
                tableSize = 1;
            } else {
                throw TableSizeMismatch();
            }

            if (param->type == hsql::kExprStar) {
                std::string alias;
                if (expr->alias) {
                    alias = expr->alias;
                } else {
                    throw UnsupportedOperationError("Alias is required for aggregate functions");
                }

                final_result_construction.result->headers.push_back(alias);
                final_result_construction.col_source.push_back({});
                std::string func_name = expr->name;
                if (StringUtils::equalsIgnoreCase(func_name, "count")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = INTEGER;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::count(result.result));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else {
                    throw UnsupportedOperationError(func_name + "(table) not supported");
                }
            } else if (param->type == hsql::kExprColumnRef) {
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
                final_result_construction.result->headers.push_back(alias);
                final_result_construction.col_source.push_back(result.col_source[idx]);

                auto col = result.result->columns[idx];

                // now we apply the actual function ..
                std::string func_name = expr->name;

                if (StringUtils::equalsIgnoreCase(func_name, "sum")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = col->type;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::sum(col));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else if (StringUtils::equalsIgnoreCase(func_name, "avg")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = FLOAT;
                    if (col->type == DateTime) final_result_construction.result->columns.back()->type = DateTime;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::avg(col));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else if (StringUtils::equalsIgnoreCase(func_name, "min")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = col->type;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::min(col));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else if (StringUtils::equalsIgnoreCase(func_name, "max")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = col->type;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::max(col));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else if (StringUtils::equalsIgnoreCase(func_name, "count")) {
                    final_result_construction.result->columns.push_back(new column());
                    final_result_construction.result->columns.back()->type = INTEGER;
                    final_result_construction.result->columns.back()->data.push_back(Agg::CPU::count(col));
                    final_result_construction.result->columns.back()->nulls.push_back(0);
                } else {
                    throw UnsupportedOperationError(func_name + "(col) not supported");
                }
            }
        } else if (expr->type == hsql::kExprSelect) {
            if (tableSize != 0 && tableSize != result.result->size()) {
                throw TableSizeMismatch();
            }

            tableSize = result.result->size();
            FromResolver::ResolveResult injectedInput;
            auto injectedTable = new table();
            injectedTable->headers = result.result->headers;
            for (auto& col: result.result->columns) {
                injectedTable->columns.push_back(new column());
                injectedTable->columns.back()->type = col->type;
                injectedTable->columns.back()->data.push_back(tval{});
                injectedTable->columns.back()->nulls.push_back(0);
            }

            injectedInput.table_names.push_back(result_tables);
            injectedInput.isTemporary.push_back(false);
            injectedInput.tables.push_back(injectedTable);

            int output_col_idx_start = -1;
            int output_col_idx_end   = -1;

            for (size_t rowIdx = 0; rowIdx < result.result->size();rowIdx++) {
                for (size_t colIdx = 0; colIdx < result.result->columns.size();colIdx++) {
                    GFI::clearCache(injectedTable->columns[colIdx]);
                    injectedTable->columns[colIdx]->nulls[0] = result.result->columns[colIdx]->nulls[rowIdx];
                    injectedTable->columns[colIdx]->data[0]  = result.result->columns[colIdx]->data[rowIdx];
                }

                auto _subResult = SelectExecutor::CPU::Execute(expr->select, tables, injectedInput);
                if (_subResult.second->size() == 0) {
                    throw UnsupportedOperationError("Excepted Sub query to have at least one row.");
                }

                if (_subResult.second->size() > 1) {
                    std::cout << "Warn, expected sub query to return one result only returned: " << fcolor(std::to_string(_subResult.second->size()), RED_FG) << std::endl;
                }

                if (_subResult.second->columns.size() != 1 && expr->alias != nullptr) {
                    throw UnsupportedOperationError("Excepted Sub query with alias to have one column");
                }

                if (expr->alias) {
                    std::string _col_name = expr->alias;
                    if (output_col_idx_start == -1) {
                        final_result_construction.result->columns.push_back(new column());
                        final_result_construction.result->headers.push_back(_col_name);
                        final_result_construction.col_source.push_back({});
                        output_col_idx_start = final_result_construction.result->headers.size() - 1;
                    }

                    final_result_construction.result->columns[output_col_idx_start]->type = _subResult.second->columns[0]->type;
                    final_result_construction.result->columns[output_col_idx_start]->data.push_back(
                            ValuesHelper::copy(
                                _subResult.second->columns[0]->data[0],
                                final_result_construction.result->columns[output_col_idx_start]->type
                                )
                            );
                    final_result_construction.result->columns[output_col_idx_start]->nulls.push_back(_subResult.second->columns[0]->nulls[0]);

                    if (_subResult.second->columns[0]->nulls[0]) {
                        final_result_construction.result->columns[output_col_idx_start]->nullsCount++;
                    }
                } else {
                    if (output_col_idx_start == -1) {
                        output_col_idx_start = final_result_construction.result->headers.size();
                        for (size_t _sub_col_size = 0; _sub_col_size < _subResult.second->columns.size();_sub_col_size++) {
                            final_result_construction.result->columns.push_back(new column());
                            final_result_construction.result->headers.push_back(_subResult.second->headers[_sub_col_size]);
                            final_result_construction.result->columns[output_col_idx_start]->type = _subResult.second->columns[_sub_col_size]->type;
                            final_result_construction.col_source.push_back({});
                        }

                        output_col_idx_end = final_result_construction.result->headers.size();
                    }

                    for (int colIdx = output_col_idx_start; colIdx < output_col_idx_end; colIdx++) {
                        final_result_construction.result->columns[colIdx]->data.push_back(
                            ValuesHelper::copy(
                                _subResult.second->columns[colIdx - output_col_idx_start]->data[0],
                                final_result_construction.result->columns[colIdx]->type
                                )
                            );

                        final_result_construction.result->columns[colIdx]->nulls.push_back(_subResult.second->columns[colIdx - output_col_idx_start]->nulls[0]);
                        if (_subResult.second->columns[colIdx - output_col_idx_start]->nulls[0]) {
                            final_result_construction.result->columns[colIdx]->nullsCount++;
                        }
                    }
                }

                delete _subResult.second;
            }

            for (auto& col: injectedTable->columns) {
                col->data.clear();
            }

            delete injectedTable;
        } else {
            throw UnsupportedOperationError("Idk wtf is this type of projection");
        }
    }


    auto orderBy = stmnt->order;
    if (orderBy && orderBy->size() > 1)
        throw UnsupportedOperationError("Order by is not supported with more than one col");

    if (orderBy && orderBy->size() == 1) {
        auto order = (*orderBy)[0];
        auto expr = order->expr;

        std::string col_name = expr->name;
        std::string table_name;

        if (expr->table != nullptr) {
            table_name = expr->table; // limit to specific table
        }

        auto idx = getColumn(&final_result_construction, table_name, col_name);
        if (idx == -1) {
            throw NoSuchColumnError(table_name + "." + col_name);
        } else if (idx == -2) {
            throw NoSuchTableError(table_name);
        }

        auto col = final_result_construction.result->columns[idx];
        if (!col->isSortIndexed()) col->buildSortedIndexes();
        auto sorted = col->sorted;

        if (order->type == hsql::OrderType::kOrderDesc) {
            for (index_t i = 0; i < sorted.size() / 2; i++) {
                auto k = sorted[i];
                sorted[i] = sorted[sorted.size() - i - 1];
                sorted[sorted.size() - i - 1] = k;
            }
        }

        auto final_final_pro_max = new table();
        final_final_pro_max->headers = final_result_construction.result->headers;

        for (const auto& _f_col: final_result_construction.result->columns) {
            final_final_pro_max->columns.push_back(new column());
            final_final_pro_max->columns.back()->data  = std::vector<tval>(_f_col->data.size());
            final_final_pro_max->columns.back()->nulls = std::vector<char>(_f_col->data.size());
            final_final_pro_max->columns.back()->type = _f_col->type;
        }

        index_t index = 0;

        for (const index_t &i : sorted) {
            for (int j = 0;j < final_final_pro_max->columns.size(); ++j) {
                final_final_pro_max->columns[j]->data[index]  = final_result_construction.result->columns[j]->data[i];
                final_final_pro_max->columns[j]->nulls[index] = final_result_construction.result->columns[j]->nulls[i];
            }

            index++;
        }

        for (const auto& _f_col: final_result_construction.result->columns) {
            _f_col->data.clear();
        }

        delete final_result_construction.result;
        final_result_construction.result = final_final_pro_max;
    }

    if (FromResolver::shouldDeleteIntermediate(global_query_input, result.result)) delete result.result;

    // clean up any temp tables created in the process
    for (int i = 0;i < global_query_input.tables.size();i++) {
        // clean up
        if (global_query_input.isTemporary[i]) {
            delete global_query_input.tables[i];
        }
    }

    return {result_tables, final_result_construction.result};
}

SelectExecutor::CPU::ConstructionResult SelectExecutor::CPU::ConstructTable(
    tensor<char, Device::CPU>* intermediate,
    const FromResolver::ResolveResult* input
    ) {
    size_t resultSize = intermediate->totalSize();

    // ReSharper disable once CppDFAMemoryLeak
    auto result = new table();
    std::vector<std::set<std::string>> col_source;

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
                    result->columns[j]->data.push_back(ValuesHelper::copy(val, t->columns[k]->type));
                    result->columns[j]->nulls.push_back(t->columns[k]->nulls[tuple_index[m]]);
                    if ( t->columns[k]->nulls[tuple_index[m]]) {
                        result->columns[j]->nullsCount++;
                    }
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

void SelectExecutor::CPU::AppendTable(
    tensor<char, Device::CPU> *intermediate,
    const FromResolver::ResolveResult *input,
    const std::vector<size_t> &offset,
    const table *result
    ) {

    std::vector<size_t> pos(offset.size(), 0);

    for (size_t i = 0; i < intermediate->totalSize(); i++) {
        if ((*intermediate)[i]) {

            auto tuple_index = intermediate->unmap(i);
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
                    result->columns[j]->data.push_back(ValuesHelper::copy(val, t->columns[k]->type));
                    result->columns[j]->nulls.push_back(t->columns[k]->nulls[pos[m]]);
                    if ( t->columns[k]->nulls[pos[m]]) {
                        result->columns[j]->nullsCount++;
                    }
                    j++;
                }
            }
        }
    }
}

int SelectExecutor::CPU::getColumn(const ConstructionResult *result, const std::string& table, const std::string& column) {
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
        std::vector<SelectExecutor::CPU::MultDimVector>& result,
        const std::vector<size_t>& inputSize,
        const std::vector<size_t>& tileSize,
        SelectExecutor::CPU::MultDimVector pos,
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

std::vector<SelectExecutor::CPU::MultDimVector> SelectExecutor::CPU::Schedule(const std::vector<size_t> &inputSize) {
    std::vector<MultDimVector> result;
    std::vector<size_t> _startPos(inputSize.size(), 0);
    std::vector<size_t> _tileSize = Cfg::getTileSizeFor(inputSize);
    ScheduleInternalRecursive(result, inputSize, _tileSize, _startPos, 0);
    return result;
}
