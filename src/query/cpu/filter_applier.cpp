//
// Created by xabdomo on 4/15/25.
//

#include "filter_applier.hpp"

#include "query/errors.hpp"

tensor<char, CPU>* FilterApplier::apply(
        std::unordered_map<std::string, table *> &tables,
        hsql::Expr *eval,
        hsql::LimitDescription *limit
    ) {
    auto expr_type = eval->type;
    if (expr_type == hsql::ExprType::kExprOperator) {
        // handle operators
        auto op = eval->opType;
        if (op == hsql::kOpAnd) {
            auto left  = eval->expr;
            auto right = eval->expr2;

        }

        throw UnsupportedOperatorError(std::to_string(op));
    } else if (expr_type == hsql::ExprType::kExprLiteralString) {

    } else {
        throw UnsupportedOperatorError(eval->getName());
    }
    return nullptr;
}
