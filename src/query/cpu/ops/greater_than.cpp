//
// Created by xabdomo on 4/16/25.
//

#include "greater_than.hpp"

#include "store.hpp"
#include "query/errors.hpp"


// #define OP_GREATER_DEBUG

tensor<char, Device::CPU> * Ops::CPU::greater_than(
    FromResolver::CPU::ResolveResult *input_data,
    hsql::Expr *left,
    hsql::Expr *right,
    hsql::LimitDescription *limit,
    const std::vector<size_t>& tile_start,
    const std::vector<size_t>& tile_size) {

    #ifdef OP_GREATER_DEBUG
    std::cout << "kExprOperator::Greater" << std::endl;
#endif
    std::vector<size_t> result_size;

    for (int i = 0;i < input_data->table_names.size();i++) {
        auto table = input_data->tables[i];
        if (table->columns.empty()) {
            return new tensor<char, Device::CPU>({});
        } else {
            if (!tile_size.empty())
                result_size.push_back(tile_size[i]);
            else
                result_size.push_back(table->columns[0]->data.size());
        }
    }

    std::vector<size_t> result_offset(result_size.size(), 0);
    if (!tile_start.empty()) result_offset = tile_start;

    auto* result = new tensor<char, Device::CPU>(result_size);
    if (left->type != hsql::kExprColumnRef && right->type != hsql::kExprColumnRef) {
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater two literals" << std::endl;
#endif
        bool ok = true;

        auto left_ = ValuesHelper::getLiteralFrom(left);
        auto right_ = ValuesHelper::getLiteralFrom(right);

        try {
            result->setAll(ValuesHelper::cmp(left_.first, right_.first, left_.second, right_.second) > 0 ? 1 : 0);
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
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater one literal" << std::endl;
#endif
        // only one of them is literal, the other is a col
        hsql::Expr* literal;
        hsql::Expr* col;
        bool literal_on_left = false;

        if (left->type != hsql::kExprColumnRef) {
            literal = left;
            col = right;
            literal_on_left = true;
        } else {
            literal = right;
            col = left;
            literal_on_left = false;
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
        if (table_idx < 0||
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

        auto literal_ = ValuesHelper::getLiteralFrom(literal);
        tval value{};
        try {
            value = ValuesHelper::castTo(literal_.first, literal_.second, column_ptr->type);
        } catch (...) {
            throw std::invalid_argument("Type mismatch between column and literal");
        }

        if (column_ptr->isSortIndexed()) {
            // if the column is sorted indexed, we can use that to speed up the search
            // we need to create a mask for the other columns
            // and a hyperplane position for the current column
            // and then we can just fill the result tensor with the matching values

#ifdef OP_GREATER_DEBUG
            std::cout << "kExprOperator::Greater Accelerating using sorted index" << std::endl;
#endif
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

            auto bucket = column_ptr->sortSearch(value, literal_on_left ? column::SST_LT : column::SST_GT);


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

            for (size_t i = result_offset[table_index]; i < result_offset[table_index] + result_size[table_index]; i++) {
                const auto& val = column_ptr->data[i];
                if (
                    (ValuesHelper::cmp(value, val , column_ptr->type) > 0 &&  literal_on_left) ||
                    (ValuesHelper::cmp(value, val , column_ptr->type) < 0 && !literal_on_left)
                    ) {
                    result->fill(1, hyperplane_pos, mask);
                    }

                hyperplane_pos[table_index]++;
            }
        }

        ValuesHelper::deleteValue(value, column_ptr->type);

        return result;
    }

#ifdef OP_GREATER_DEBUG
    std::cout << "kExprOperator::Greater no literal" << std::endl;
#endif
    // both are cols
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

    // now we have the two columns
    // we need to implement the search logic
    if (!column_ptr_l->isSortIndexed() && !column_ptr_r->isSortIndexed()) {
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater no sorted index, building ... ";
#endif
        if (column_ptr_r->data.size() > column_ptr_l->data.size()) {
            column_ptr_r->buildSortedIndexes();
        } else {
            column_ptr_l->buildSortedIndexes();
        }
#ifdef OP_GREATER_DEBUG
        std::cout << "Done" << std::endl;;
#endif
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
    result->setAll(0);

    // now do the search
    auto sorted = column_ptr_r;
    auto other = column_ptr_l;
    auto hashed_index = table_index_r;
    auto other_index = table_index_l;
    if (!sorted->isSortIndexed()) {
        sorted = column_ptr_l;
        other = column_ptr_r;
        hashed_index = table_index_l;
        other_index = table_index_r;
    }

    for (int i = result_offset[other_index];i < result_offset[other_index] + result_size[other_index];i++) {
        auto matches = sorted->sortSearch(other->data[i], other_index == table_index_l ? column::SST_LT : column::SST_GT);
        hyperplane_pos[other_index] = i - result_offset[other_index];
        for (auto match: matches) {
            if (match < result_offset[hashed_index] || match >= result_offset[hashed_index] + result_size[hashed_index]) {
                continue;
            }

            hyperplane_pos[hashed_index] = match;
            result->fill(1, hyperplane_pos, mask);
        }
    }

    return result;
}
