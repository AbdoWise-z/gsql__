//
// Created by xabdomo on 4/15/25.
//

#include "filter_applier.hpp"

#include "query/errors.hpp"

#define FILTER_DEBUG

tensor<char, CPU>* FilterApplier::apply(
        FromResolver::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit
    ) {

    if (eval == nullptr) { // pass all
        std::vector<size_t> literal_sizes;

        for (int i = 0;i < input_data->table_names.size();i++) {
            auto table = input_data->tables[i];
            if (table->columns.empty()) {
                return new tensor<char, CPU>({});
            } else {
                literal_sizes.push_back(table->columns[0].data.size());
            }
        }

        auto* result = new tensor<char, CPU>(literal_sizes);
        result->setAll(1);
        return result;
    }

    auto expr_type = eval->type;
    if (expr_type == hsql::ExprType::kExprOperator) {
        // handle operators
        auto op = eval->opType;
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprOperator" << op << std::endl;
#endif
        if (op == hsql::kOpAnd) {
#ifdef FILTER_DEBUG
            std::cout << "kExprOperator::AND" << std::endl;
#endif
            auto left  = eval->expr;
            auto right = eval->expr2;
            // set limit to always be nullptr because
            // count(left) AND count(right) <= count(this)
            // so I cannot limit either left or right since
            // I may undershoot the limit this way.
            const auto left_result  = FilterApplier::apply(input_data, left,  nullptr);
            const auto right_result = FilterApplier::apply(input_data, right, nullptr);

            auto result = new tensor(*left_result && *right_result);

            delete left_result;
            delete right_result;


            return result;
        }

        if (op == hsql::kOpEquals) {
#ifdef FILTER_DEBUG
            std::cout << "kExprOperator::Equals" << std::endl;
#endif
            auto left  = eval->expr;
            auto right = eval->expr2;

            std::vector<size_t> literal_sizes;

            for (const auto& table: input_data->tables) {
                if (table->columns.empty()) {
                    return new tensor<char, CPU>({});
                }
                literal_sizes.push_back(table->columns[0].data.size());
            }

            auto* result = new tensor<char, CPU>(literal_sizes);

            if (left->isLiteral() && right->isLiteral()) {

#ifdef FILTER_DEBUG
                std::cout << "kExprOperator::Equals two literals" << std::endl;
#endif
                // both are literal, just check if they are equal
                if (left->type != right->type) {
                    // we know they can't be equal if they can't be the same type (unless they are float and int)
                    // fixme: handle case where they are float or int
                    result->setAll(0);
#ifdef FILTER_DEBUG
                    std::cout << "kExprOperator::Equals type mismatch" << std::endl;
#endif
                    return result;
                }

                if (left->type == hsql::ExprType::kExprLiteralString) {
#ifdef FILTER_DEBUG
                    std::cout << "kExprOperator::Equals String, left=" << left->name << " & right=" << right->name << std::endl;
#endif
                    auto r = strcmp(left->name, right->name);
                    result->setAll(r == 0 ? 1 : 0);
                    return result;
                }

                if (left->type == hsql::ExprType::kExprLiteralInt) {
#ifdef FILTER_DEBUG
                    std::cout << "kExprOperator::Equals Integer, left=" << left->ival << " & right=" << right->ival << std::endl;
#endif
                    result->setAll(left->ival == right->ival);
                    return result;
                }

                if (left->type == hsql::ExprType::kExprLiteralFloat) {
#ifdef FILTER_DEBUG
                    std::cout << "kExprOperator::Equals Float, left=" << left->fval << " & right=" << right->fval << std::endl;
#endif
                    result->setAll(left->fval == right->fval);
                    return result;
                }

                throw UnsupportedLiteralError();
            }

            if (left->isLiteral() || right->isLiteral()) {
#ifdef FILTER_DEBUG
                std::cout << "kExprOperator::Equals one literal" << std::endl;
#endif
                // only one of them is literal, the other is a col
                hsql::Expr* literal;
                hsql::Expr* col;

                if (left->isLiteral()) {
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
                    if (tables.size() != 1) {
                        throw std::invalid_argument("Cannot filter on item without it's table name");
                    }
                    table_name = *input_data->table_names[0].begin(); // just take the first alias of the first table
                } else {
                    table_name = col->table;
                }

                // now iterate over that and do the actual comparison
                if (!tables.contains(table_name) ||
                    std::find(
                        tables[table_name]->headers.begin(),
                        tables[table_name]->headers.end(),
                        col_name
                        ) == tables[table_name]->headers.end()) {
                    throw std::invalid_argument(table_name + "." + col_name + " doesn't exist");
                }

                auto table_ptr = tables[table_name];
                ptrdiff_t pos = std::find(table_ptr->headers.begin(), table_ptr->headers.end(), col_name) - table_ptr->headers.begin();
                auto column_ptr = &table_ptr->columns[pos];

                if (
                    (column_ptr->type == STRING && literal->type != hsql::ExprType::kExprLiteralString) ||
                    (column_ptr->type == INTEGER && literal->type != hsql::ExprType::kExprLiteralInt) ||
                    (column_ptr->type == FLOAT && literal->type != hsql::ExprType::kExprLiteralFloat)
                    ) {
                    throw std::invalid_argument("Type mismatch between column and literal");
                }

                if (column_ptr->isHashIndexed()) {
                    // if the column is hash indexed, we can use that to speed up the search
                    // we need to create a mask for the other columns
                    // and a hyperplane position for the current column
                    // and then we can just fill the result tensor with the matching values

#ifdef FILTER_DEBUG
                    std::cout << "kExprOperator::Equals Accelerating using hash index" << std::endl;
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

                    result->setAll(0);

                    tval value;
                    if (column_ptr->type == STRING)
                        value = create_from(literal->name);
                    else if (column_ptr->type == INTEGER)
                        value = create_from(literal->ival);
                    else
                        value = create_from(literal->fval);

                    auto bucket = column_ptr->hashSearch(value);
                    deleteValue(value, column_ptr->type);

                    for (auto r: bucket) {
                        hyperplane_pos[table_index] = r;
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

                    result->setAll(0);

                    if (column_ptr->type == STRING) {
#ifdef FILTER_DEBUG
                        std::cout << "kExprOperator::Equals literal_val(str)=" << literal->name << std::endl;
#endif
                        for (auto val: column_ptr->data) {
                            if (strcmp(val.s->c_str(), literal->name) == 0) {
                                result->fill(1, hyperplane_pos, mask);
                            }

                            hyperplane_pos[table_index]++;
                        }
                    } else if (column_ptr->type == INTEGER) {
#ifdef FILTER_DEBUG
                        std::cout << "kExprOperator::Equals literal_val(int)=" << literal->ival << std::endl;
#endif
                        for (auto val: column_ptr->data) {
                            if (val.i == literal->ival) {
                                result->fill(1, hyperplane_pos, mask);
                            }

                            hyperplane_pos[table_index]++;
                        }
                    } else {
#ifdef FILTER_DEBUG
                        std::cout << "kExprOperator::Equals literal_val(float)=" << literal->fval << std::endl;
#endif
                        for (auto val: column_ptr->data) {
                            if (val.d == literal->fval) {
                                result->fill(1, hyperplane_pos, mask);
                            }

                            hyperplane_pos[table_index]++;
                        }
                    }
                }

                return result;
            }

#ifdef FILTER_DEBUG
            std::cout << "kExprOperator::Equals no literal" << std::endl;
#endif
            // both are cols
            // well now we really need those hashes ..
            std::string col_name_l   = left->name;
            std::string table_name_l ;

            std::string col_name_r   = right->name;
            std::string table_name_r ;

            if (left->table == nullptr) {
                if (tables.size() != 1) {
                    throw std::invalid_argument("Cannot filter on item without it's table name");
                }
                table_name_l = *input_data->table_names[0].begin();
            } else {
                table_name_l = left->table;
            }

            if (right->table == nullptr) {
                if (tables.size() != 1) {
                    throw std::invalid_argument("Cannot filter on item without it's table name");
                }
                table_name_r = *input_data->table_names[0].begin();
            } else {
                table_name_r = right->table;
            }

            if (!tables.contains(table_name_l) ||
                    std::find(
                        tables[table_name_l]->headers.begin(),
                        tables[table_name_l]->headers.end(),
                        col_name_l
                        ) == tables[table_name_l]->headers.end()
                        )
            {
                throw std::invalid_argument(table_name_l + "." + col_name_l + " doesn't exist");
            }

            auto table_ptr_l = tables[table_name_l];
            ptrdiff_t pos_l = std::find(table_ptr_l->headers.begin(), table_ptr_l->headers.end(), col_name_l) - table_ptr_l->headers.begin();
            auto column_ptr_l = &table_ptr_l->columns[pos_l];

            if (!tables.contains(table_name_r) ||
                    std::find(
                        tables[table_name_r]->headers.begin(),
                        tables[table_name_r]->headers.end(),
                        col_name_r
                        ) == tables[table_name_r]->headers.end()
                        )
            {
                throw std::invalid_argument(table_name_r + "." + col_name_r + " doesn't exist");
            }

            auto table_ptr_r = tables[table_name_r];
            ptrdiff_t pos_r = std::find(table_ptr_r->headers.begin(), table_ptr_r->headers.end(), col_name_r) - table_ptr_r->headers.begin();
            auto column_ptr_r = &table_ptr_r->columns[pos_r];

            if (column_ptr_r->type != column_ptr_l->type) {
                throw std::invalid_argument("Type mismatch between two columns");
            }

            // now we have the two columns
            // we need to implement the search logic
            if (!column_ptr_l->isHashIndexed() && !column_ptr_r->isHashIndexed()) {
#ifdef FILTER_DEBUG
                std::cout << "kExprOperator::Equals no hashes, building ..." << std::endl;
#endif
                if (column_ptr_r->data.size() > column_ptr_l->data.size()) {
                    column_ptr_r->buildHashedIndexes(Cfg::HashTableExtendableSize);
                } else {
                    column_ptr_l->buildHashedIndexes(Cfg::HashTableExtendableSize);
                }
#ifdef FILTER_DEBUG
                std::cout << "kExprOperator::Equals no hashes, done ..." << std::endl;
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

            for (int i = 0;i < other->data.size();i++) {
                auto matches = hashed->hashSearch(other->data[i]);
                hyperplane_pos[other_index] = i;
                for (auto match: matches) {
                    hyperplane_pos[hashed_index] = match;
                    result->fill(1, hyperplane_pos, mask);
                }
            }

            return result;
        }

        throw UnsupportedOperatorError(std::to_string(op));
    } else if (expr_type == hsql::ExprType::kExprLiteralString) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralString" << std::endl;
#endif
        std::vector<size_t> literal_sizes;

        for (const auto& table: input_data->tables) {
            if (table->columns.empty()) {
                return new tensor<char, CPU>({});
            } else {
                literal_sizes.push_back(table->columns[0].data.size());
            }
        }

        auto* result = new tensor<char, CPU>(literal_sizes);
        result->setAll(strlen(eval->name) > 0 ? 1 : 0);
        return result;
    } else if (expr_type == hsql::ExprType::kExprLiteralInt) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralInt" << std::endl;
#endif
        std::vector<size_t> literal_sizes;

        for (const auto& table: input_data->tables) {
            if (table->columns.empty()) {
                return new tensor<char, CPU>({});
            } else {
                literal_sizes.push_back(table->columns[0].data.size());
            }
        }


        auto* result = new tensor<char, CPU>(literal_sizes);
        result->setAll(eval->ival > 0 ? 1 : 0);
        return result;
    } else if (expr_type == hsql::ExprType::kExprLiteralFloat) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralFloat" << std::endl;
#endif

        std::vector<size_t> literal_sizes;

        for (const auto& table: input_data->tables) {
            if (table->columns.empty()) {
                return new tensor<char, CPU>({});
            } else {
                literal_sizes.push_back(table->columns[0].data.size());
            }
        }

        auto* result = new tensor<char, CPU>(literal_sizes);
        result->setAll(eval->fval > 0 ? 1 : 0);
        return result;
    } else {
        throw UnsupportedOperatorError(eval->getName());
    }
    return nullptr;
}
