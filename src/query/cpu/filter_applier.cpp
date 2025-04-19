//
// Created by xabdomo on 4/15/25.
//

#include "filter_applier.hpp"

#include "ops/equality.hpp"
#include "ops/greater_than.hpp"
#include "ops/logical_and.hpp"
#include "ops/logical_or.hpp"
#include "query/errors.hpp"

// #define FILTER_DEBUG

tensor<char, CPU>* FilterApplier::apply(
        FromResolver::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start,
        const std::vector<size_t>& tile_size
    ) {

    std::vector<size_t> result_size;

    for (int i = 0;i < input_data->table_names.size();i++) {
        auto table = input_data->tables[i];
        if (table->columns.empty()) {
            return new tensor<char, CPU>({});
        } else {
            if (!tile_size.empty())
                result_size.push_back(tile_size[i]);
            else
                result_size.push_back(table->columns[0]->data.size());
        }
    }

    if (eval == nullptr) { // pass all
        auto* result = new tensor<char, CPU>(result_size);
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
            return Ops::logical_and(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpOr) {
            return Ops::logical_or(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpEquals) {
            return Ops::equality(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpGreater) {
            return Ops::greater_than(input_data, eval->expr, eval->expr2, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpLess) {
            return Ops::greater_than(input_data, eval->expr2, eval->expr, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpGreaterEq) {
            auto t1 = Ops::greater_than(input_data, eval->expr, eval->expr2, limit, tile_start, tile_size);
            auto t2 = Ops::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor(*t1 || *t2);
            delete t1;
            delete t2;
            return result; // yes I am lazy sue me ig.
        }

        if (op == hsql::kOpLessEq) {
            auto t1 = Ops::greater_than(input_data, eval->expr2, eval->expr, limit, tile_start, tile_size);
            auto t2 = Ops::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor(*t1 || *t2);
            delete t1;
            delete t2;
            return result;
        }

        if (op == hsql::kOpNotEquals) {
            auto t2 = Ops::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor(!*t2);
            delete t2;
            return result;
        }

        throw UnsupportedOperatorError(std::to_string(op));
    } else if (expr_type == hsql::ExprType::kExprLiteralString) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralString" << std::endl;
#endif

        auto* result = new tensor<char, CPU>(result_size);
        result->setAll(strlen(eval->name) > 0 ? 1 : 0);
        return result;
    } else if (expr_type == hsql::ExprType::kExprLiteralInt) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralInt" << std::endl;
#endif

        auto* result = new tensor<char, CPU>(result_size);
        result->setAll(eval->ival > 0 ? 1 : 0);
        return result;
    } else if (expr_type == hsql::ExprType::kExprLiteralFloat) {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprLiteralFloat" << std::endl;
#endif

        auto* result = new tensor<char, CPU>(result_size);
        result->setAll(eval->fval > 0 ? 1 : 0);
        return result;
    } else {
        throw UnsupportedOperatorError(eval->getName());
    }
    return nullptr;
}
