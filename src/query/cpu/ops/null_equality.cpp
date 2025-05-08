//
// Created by xabdomo on 4/16/25.
//

#include "null_equality.hpp"

#include "store.hpp"

// #define OP_EQUALS_DEBUG

tensor<char, Device::CPU> * Ops::CPU::null_equality(
    FromResolver::CPU::ResolveResult *input_data,
    hsql::Expr *eval,
    hsql::LimitDescription *limit,
    const std::vector<size_t>& tile_start,
    const std::vector<size_t>& tile_size) {
#ifdef OP_EQUALS_DEBUG
    std::cout << "kExprOperator::Equals" << std::endl;
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

    result->setAll(0);

    auto col = eval->expr;

    if (col->type != hsql::kExprColumnRef) {
#ifdef OP_EQUALS_DEBUG
        std::cout << "kExprOperator::null_equality one literal" << std::endl;
#endif
        result->setAll(1);
        return result;
    }

#ifdef OP_EQUALS_DEBUG
    std::cout << "kExprOperator::null_equality no literal" << std::endl;
#endif

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

    for (int i = result_offset[table_index];i < result_offset[table_index] + result_size[table_index];i++) {
        if (!column_ptr->nulls[i]) continue;
         hyperplane_pos[table_index] = i - result_offset[table_index];
        result->fill(1, hyperplane_pos, mask);
    }

    return result;
}
