//
// Created by xabdomo on 4/16/25.
//

#include "greater_than.hpp"

#include "store.hpp"
#include "query/errors.hpp"


// #define OP_GREATER_DEBUG

tensor<char, Device::GPU> * Ops::GPU::greater_than(
    FromResolver::GPU::ResolveResult *input_data,
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
    result->setAll(0);

    if (left->isLiteral() && right->isLiteral()) {
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater two literals" << std::endl;
#endif
                // both are literal, just check if they are equal
        if (left->type != right->type) {
            // we know they can't be equal if they can't be the same type (unless they are float and int)
            // fixme: handle case where they are float or int
#ifdef OP_GREATER_DEBUG
            std::cout << "kExprOperator::Greater type mismatch" << std::endl;
#endif
            return result;
        }

        if (left->type == hsql::ExprType::kExprLiteralString) {
#ifdef OP_GREATER_DEBUG
            std::cout << "kExprOperator::Greater String, left=" << left->name << " & right=" << right->name << std::endl;
#endif
            int r = 0;
            auto lt = ValuesHelper::parseDateTime(left->name);
            auto rt = ValuesHelper::parseDateTime(left->name);
            if (lt != std::nullopt && rt != std::nullopt) {
                r = ValuesHelper::cmp(rt.value(), rt.value());
            } else {
                r = ValuesHelper::cmp(left->name, right->name);
            }

            result->setAll(r > 0 ? 1 : 0);
            return result;
        }

        if (left->type == hsql::ExprType::kExprLiteralInt) {
#ifdef OP_GREATER_DEBUG
            std::cout << "kExprOperator::Greater Integer, left=" << left->ival << " & right=" << right->ival << std::endl;
#endif
            result->setAll(left->ival > right->ival);
            return result;
        }

        if (left->type == hsql::ExprType::kExprLiteralFloat) {
#ifdef OP_GREATER_DEBUG
            std::cout << "kExprOperator::Greater Float, left=" << left->fval << " & right=" << right->fval << std::endl;
#endif
            result->setAll(left->fval > right->fval);
            return result;
        }

        throw UnsupportedLiteralError();
    }

    if (left->isLiteral() || right->isLiteral()) {
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater one literal" << std::endl;
#endif
        // only one of them is literal, the other is a col
        hsql::Expr* literal;
        hsql::Expr* col;
        bool literal_on_left = false;

        if (left->isLiteral()) {
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

        auto table_idx = FromResolver::GPU::find(input_data, table_name);
        if (table_idx == -1 ||
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

        if ((column_ptr->type == STRING && literal->type != hsql::ExprType::kExprLiteralString) ||
            (column_ptr->type == INTEGER && literal->type != hsql::ExprType::kExprLiteralInt) ||
            (column_ptr->type == FLOAT && literal->type != hsql::ExprType::kExprLiteralFloat)
            ) {
            if (column_ptr->type == DateTime && literal->type != hsql::ExprType::kExprLiteralDate) {

            }
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

            tval value;
            if (column_ptr->type == STRING)
                value = ValuesHelper::create_from(literal->name);
            else if (column_ptr->type == INTEGER)
                value = ValuesHelper::create_from(literal->ival);
            else
                value = ValuesHelper::create_from(literal->fval);

            auto bucket = column_ptr->sortSearch(value, literal_on_left ? column::SST_GT : column::SST_LT);
            ValuesHelper::deleteValue(value, column_ptr->type);

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

            if (column_ptr->type == STRING) {
#ifdef OP_GREATER_DEBUG
                std::cout << "kExprOperator::Greater literal_val(str)=" << literal->name << std::endl;
#endif
                for (int i = result_offset[table_index]; i < result_offset[table_index] + result_size[table_index]; i++) {
                    auto val = column_ptr->data[i];
                    if (
                        (strcmp(literal->name, val.s->c_str()) > 0 &&  literal_on_left) ||
                        (strcmp(literal->name, val.s->c_str()) < 0 && !literal_on_left)
                        ) {
                        result->fill(1, hyperplane_pos, mask);
                    }

                    hyperplane_pos[table_index]++;
                }
            } else if (column_ptr->type == INTEGER) {
#ifdef OP_GREATER_DEBUG
                std::cout << "kExprOperator::Greater literal_val(int)=" << literal->ival << std::endl;
#endif
                for (int i = result_offset[table_index]; i < result_offset[table_index] + result_size[table_index]; i++) {
                    auto val = column_ptr->data[i];
                    if (
                        (literal->ival > val.i &&  literal_on_left) ||
                        (literal->ival < val.i && !literal_on_left)
                    ) {
                        result->fill(1, hyperplane_pos, mask);
                    }

                    hyperplane_pos[table_index]++;
                }
            } else {
#ifdef OP_GREATER_DEBUG
                std::cout << "kExprOperator::Greater literal_val(float)=" << literal->fval << std::endl;
#endif
                for (int i = result_offset[table_index]; i < result_offset[table_index] + result_size[table_index]; i++) {
                    auto val = column_ptr->data[i];
                    if (
                        (literal->fval > val.d &&  literal_on_left) ||
                        (literal->fval < val.d && !literal_on_left)
                    ) {
                        result->fill(1, hyperplane_pos, mask);
                    }

                    hyperplane_pos[table_index]++;
                }
            }
        }

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

    auto table_idx = FromResolver::GPU::find(input_data, table_name_l);
    if (table_idx == -1 ||
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

    table_idx = FromResolver::GPU::find(input_data, table_name_r);
    if (table_idx == -1 ||
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
    if (!column_ptr_l->isSortIndexed() && !column_ptr_r->isSortIndexed()) {
#ifdef OP_GREATER_DEBUG
        std::cout << "kExprOperator::Greater no sorted index, using 2D ";
#endif
        GFI::inequality(
            result,
            column_ptr_l,
            column_ptr_r,
            tile_start,
            tile_size,
            table_index_l,
            table_index_r,
            mask,
            column::SST_GT
        );

        return result;
    } else {
        // now do the search
        auto sorted = column_ptr_r;
        auto other = column_ptr_l;
        auto sorted_index = table_index_r;
        auto other_index = table_index_l;
        if (!sorted->isSortIndexed()) {
            sorted = column_ptr_l;
            other = column_ptr_r;
            sorted_index = table_index_l;
            other_index = table_index_r;
        }

        GFI::inequality(
            result,
            other,
            sorted,
            tile_start,
            tile_size,
            other_index,
            sorted_index,
            mask,
            other_index == table_index_l ? column::SST_GT : column::SST_LT
        );

        return result;
    }
}
