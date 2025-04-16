//
// Created by xabdomo on 4/15/25.
//

#include "filter_applier.hpp"

#include "ops/equality.hpp"
#include "ops/greater_than.hpp"
#include "ops/logical_and.hpp"
#include "ops/logical_or.hpp"
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
                literal_sizes.push_back(table->columns[0]->data.size());
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
            return Ops::logical_and(input_data, eval, limit);
        }

        if (op == hsql::kOpOr) {
            return Ops::logical_or(input_data, eval, limit);
        }

        if (op == hsql::kOpEquals) {
            return Ops::equality(input_data, eval, limit);
        }

        if (op == hsql::kOpGreater) {
            return Ops::greater_than(input_data, eval->expr, eval->expr2, limit);
        }

        if (op == hsql::kOpLess) {
            return Ops::greater_than(input_data, eval->expr2, eval->expr, limit);
        }

        if (op == hsql::kOpGreaterEq) {
            auto t1 = Ops::greater_than(input_data, eval->expr, eval->expr2, limit);
            auto t2 = Ops::equality(input_data, eval, limit);
            auto result = new tensor(*t1 || *t2);
            delete t1;
            delete t2;
            return result; // yes I am lazy sue me ig.
        }

        if (op == hsql::kOpLessEq) {
            auto t1 = Ops::greater_than(input_data, eval->expr2, eval->expr, limit);
            auto t2 = Ops::equality(input_data, eval, limit);
            auto result = new tensor(*t1 || *t2);
            delete t1;
            delete t2;
            return result;
        }

        if (op == hsql::kOpNotEquals) {
            auto t2 = Ops::equality(input_data, eval, limit);
            auto result = new tensor(!*t2);
            delete t2;
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
                literal_sizes.push_back(table->columns[0]->data.size());
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
                literal_sizes.push_back(table->columns[0]->data.size());
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
                literal_sizes.push_back(table->columns[0]->data.size());
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
