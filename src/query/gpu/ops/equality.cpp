//
// Created by xabdomo on 4/16/25.
//

#include "equality.hpp"

#include "store.hpp"
#include "db/value_helper.hpp"
#include "query/errors.hpp"
#include "query/gpu/gpu_function_interface.cuh"
#include "query/gpu/select_executor.hpp"

// #define OP_EQUALS_DEBUG

tensor<char, Device::GPU> * Ops::GPU::equality(
    FromResolver::GPU::ResolveResult *input_data,
    hsql::Expr *eval,
    hsql::LimitDescription *limit,
    const std::vector<size_t>& tile_start,
    const std::vector<size_t>& tile_size) {
#ifdef OP_EQUALS_DEBUG
    std::cout << "kExprOperator::Equals" << std::endl;
#endif
    auto left  = eval->expr;
    auto right = eval->expr2;

    std::vector<size_t> result_size;

    for (int i = 0;i < input_data->table_names.size();i++) {
        auto table = input_data->tables[i];
        if (table->columns.empty()) {
            return new tensor<char, Device::GPU>({});
        } else {
            if (!tile_size.empty())
                result_size.push_back(tile_size[i]);
            else
                result_size.push_back(table->columns[0]->data.size());
        }
    }

    std::vector<size_t> result_offset(result_size.size(), 0);
    if (!tile_start.empty()) result_offset = tile_start;

    auto* result = new tensor<char, Device::GPU>(result_size);
    if (left->type != hsql::kExprColumnRef && right->type != hsql::kExprColumnRef) {
#ifdef OP_EQUALS_DEBUG
        std::cout << "kExprOperator::Equals two literals" << std::endl;
#endif
        // both are literal, just check if they are equal
        bool ok = true;

        auto left_  = ValuesHelper::getLiteralFrom(left, false, SelectExecutor::GPU::global_input, ValuesHelper::GPU);
        auto right_ = ValuesHelper::getLiteralFrom(right, false, SelectExecutor::GPU::global_input, ValuesHelper::GPU);

        try {
            result->setAll(ValuesHelper::cmp(left_.first, right_.first, left_.second, right_.second) == 0 ? 1 : 0);
        } catch (...) {
            ok = false;
        }

        ValuesHelper::deleteValue(left_.first, left_.second);
        ValuesHelper::deleteValue(right_.first, right_.second);

        if (!ok) {
            throw std::invalid_argument("Type mismatch between two literals");
        }

        return result;
    }

    result->setAll(0);

    if (left->type != hsql::kExprColumnRef || right->type != hsql::kExprColumnRef) {
#ifdef OP_EQUALS_DEBUG
        std::cout << "kExprOperator::Equals one literal" << std::endl;
#endif
        // only one of them is literal, the other is a col
        hsql::Expr* literal;
        hsql::Expr* col;

        if (left->type != hsql::kExprColumnRef) {
            literal = left;
            col = right;
        } else {
            literal = right;
            col = left;
        }

        // now we get the actual column reference from the table.
        std::string col_name = col->name;
        std::string table_name;
        if (col->table == nullptr) {
            if (input_data->tables.size() != 1) {
                throw std::invalid_argument("Cannot filter on item without it's table name");
            }
            table_name = *input_data->table_names[0].begin(); // just take the first alias of the first table
        } else {
            table_name = col->table;
        }

        auto table_idx = FromResolver::find(input_data, table_name);
        if (table_idx < 0 ||
            std::find(
                input_data->tables[table_idx]->headers.begin(),
                input_data->tables[table_idx]->headers.end(),
                col_name
                ) == input_data->tables[table_idx]->headers.end()) {
            throw std::invalid_argument(table_name + "." + col_name + " doesn't exist");
                }

        auto table_ptr = input_data->tables[table_idx];
        ptrdiff_t pos = std::find(table_ptr->headers.begin(), table_ptr->headers.end(), col_name) - table_ptr->headers.begin();
        auto column_ptr = table_ptr->columns[pos];


        tval value{};
        int dt_search_type = 0; // normal equality
        if (literal->type == hsql::kExprLiteralString) {
            auto literal_ = ValuesHelper::getLiteralFrom(literal, true);
            try {
                value = ValuesHelper::castTo(literal_.first, literal_.second, column_ptr->type);
                ValuesHelper::deleteValue(literal_.first, literal_.second);
            } catch (...) {
                if (literal_.second == STRING && column_ptr->type == DateTime) {
                    auto _test = ValuesHelper::parseDateTimeDateOnly(*literal_.first.s);
                    if (_test != std::nullopt) {
                        value = ValuesHelper::create_from(*_test);
                        dt_search_type = 1; // search only on date
                    } else {
                        _test = ValuesHelper::parseDateTimeTimeOnly(*literal_.first.s);
                        if (_test != std::nullopt) {
                            value = ValuesHelper::create_from(*_test);
                            dt_search_type = 2; // search only on time
                        } else {
                            throw std::invalid_argument("Type mismatch between column and literal");
                        }
                    }
                    ValuesHelper::deleteValue(literal_.first, literal_.second);
                } else {
                    throw std::invalid_argument("Type mismatch between column and literal");
                }
            }
        } else {
            auto literal_ = ValuesHelper::getLiteralFrom(literal, false, SelectExecutor::GPU::global_input, ValuesHelper::GPU);
            try {
                value = ValuesHelper::castTo(literal_.first, literal_.second, column_ptr->type);
                ValuesHelper::deleteValue(literal_.first, literal_.second);
            } catch (...) {
                throw std::invalid_argument("Type mismatch between column and literal");
            }
        }

        std::vector<uint64_t> mask;
        std::vector<size_t> hyperplane_pos;
        int table_index = 0;
        for (int i = 0;i < input_data->table_names.size(); ++i) {
            hyperplane_pos.push_back(0);
            if (input_data->table_names[i].contains(table_name)) {
                mask.push_back(0);
                table_index = i;
            } else {
                mask.push_back(1);
            }
        }

        if (column_ptr->isHashIndexed() && dt_search_type == 0) {
            // if the column is hash indexed, we can use that to speed up the search
            // we need to create a mask for the other columns
            // and a hyperplane position for the current column
            // and then we can just fill the result tensor with the matching values

#ifdef OP_EQUALS_DEBUG
            std::cout << "kExprOperator::Equals Accelerating using hash index" << std::endl;
#endif

            auto bucket = column_ptr->hashSearch(value);

            for (auto r: bucket) {
                if (r < result_offset[table_index] || r >= result_offset[table_index] + result_size[table_index]) {
                    // we need to check if the index is in the current tile
                    // if not, we skip it
                    continue;
                }

                hyperplane_pos[table_index] = r - result_offset[table_index];
                result->fill(1, hyperplane_pos, mask);
            }

        } else {
            if (dt_search_type == 0) { // identical
                GFI::equality(
                    result,
                    column_ptr,
                    value,
                    false,
                    tile_start,
                    tile_size,
                    table_index,
                    mask
                );
            } else if (dt_search_type == 1) { // date only
                GFI::equality_date(
                    result,
                    column_ptr,
                    value,
                    tile_start,
                    tile_size,
                    table_index,
                    mask
                );
            } else if (dt_search_type == 2) { // time only
                GFI::equality_time(
                    result,
                    column_ptr,
                    value,
                    tile_start,
                    tile_size,
                    table_index,
                    mask
                );
            }
        }

        ValuesHelper::deleteValue(value, column_ptr->type);
        return result;
    }

#ifdef OP_EQUALS_DEBUG
    std::cout << "kExprOperator::Equals no literal" << std::endl;
#endif
    // both are cols
    // well now we really need those hashes ..
    std::string col_name_l   = left->name;
    std::string table_name_l ;

    std::string col_name_r   = right->name;
    std::string table_name_r ;

    if (left->table == nullptr) {
        if (input_data->tables.size() != 1) {
            throw std::invalid_argument("Cannot filter on item without it's table name");
        }
        table_name_l = *input_data->table_names[0].begin();
    } else {
        table_name_l = left->table;
    }

    if (right->table == nullptr) {
        if (input_data->tables.size() != 1) {
            throw std::invalid_argument("Cannot filter on item without it's table name");
        }
        table_name_r = *input_data->table_names[0].begin();
    } else {
        table_name_r = right->table;
    }

    auto table_idx = FromResolver::find(input_data, table_name_l);
    if (table_idx < 0 ||
            std::find(
                input_data->tables[table_idx]->headers.begin(),
                input_data->tables[table_idx]->headers.end(),
                col_name_l
                ) == input_data->tables[table_idx]->headers.end())
    {
        throw std::invalid_argument(table_name_l + "." + col_name_l + " doesn't exist");
    }

    auto table_ptr_l = input_data->tables[table_idx];
    ptrdiff_t pos_l = std::find(table_ptr_l->headers.begin(), table_ptr_l->headers.end(), col_name_l) - table_ptr_l->headers.begin();
    auto column_ptr_l = table_ptr_l->columns[pos_l];

    table_idx = FromResolver::find(input_data, table_name_r);
    if (table_idx < 0 ||
            std::find(
                input_data->tables[table_idx]->headers.begin(),
                input_data->tables[table_idx]->headers.end(),
                col_name_r
                ) == input_data->tables[table_idx]->headers.end())
    {
        throw std::invalid_argument(table_name_r + "." + col_name_r + " doesn't exist");
    }

    auto table_ptr_r = input_data->tables[table_idx];
    ptrdiff_t pos_r = std::find(table_ptr_r->headers.begin(), table_ptr_r->headers.end(), col_name_r) - table_ptr_r->headers.begin();
    auto column_ptr_r = table_ptr_r->columns[pos_r];

    if (column_ptr_r->type != column_ptr_l->type) {
        throw std::invalid_argument("Type mismatch between two columns");
    }

    std::vector<uint64_t> mask;
    std::vector<size_t> hyperplane_pos;
    int table_index_l = 0;
    int table_index_r = 0;
    for (int i = 0;i < input_data->table_names.size(); ++i) {
        hyperplane_pos.push_back(0);
        if (input_data->table_names[i].contains(table_name_l)) {
            mask.push_back(0);
            table_index_l = i;
        } else if (input_data->table_names[i].contains(table_name_r)) {
            mask.push_back(0);
            table_index_r = i;
        } else {
            mask.push_back(1);
        }
    }

    // now we have the two columns
    // we need to implement the search logic
    if (!column_ptr_l->isHashIndexed() && !column_ptr_r->isHashIndexed()) {
#ifdef OP_EQUALS_DEBUG
        std::cout << "kExprOperator::Equals no hashes, using 2D ";
#endif
        GFI::equality(
            result,
            column_ptr_l,
            column_ptr_r,
            tile_start,
            tile_size,
            table_index_l,
            table_index_r,
            mask
        );

        return result;
    } else {
#ifdef OP_EQUALS_DEBUG
        std::cout << "kExprOperator::Equals hashes found .. using it. ";
#endif

        auto hashed = column_ptr_r;
        auto other = column_ptr_l;
        auto hashed_index = table_index_r;
        auto other_index = table_index_l;
        if (!hashed->isHashIndexed()) {
            hashed = column_ptr_l;
            other = column_ptr_r;
            hashed_index = table_index_l;
            other_index = table_index_r;
        }

        GFI::equality(
            result,
            other,
            hashed,
            tile_start,
            tile_size,
            other_index,
            hashed_index,
            mask
        );

        // for (int i = result_offset[other_index];i < result_offset[other_index] + result_size[other_index];i++) {
        //     auto matches = hashed->hashSearch(other->data[i]);
        //     hyperplane_pos[other_index] = i - result_offset[other_index];
        //     for (auto match: matches) {
        //         if (match < result_offset[hashed_index] || match >= result_offset[hashed_index] + result_size[hashed_index]) {
        //             continue;
        //         }
        //
        //         hyperplane_pos[hashed_index] = match - result_offset[hashed_index];
        //         result->fill(1, hyperplane_pos, mask);
        //     }
        // }

        return result;
    }
}
