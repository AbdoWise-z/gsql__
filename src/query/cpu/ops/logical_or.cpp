//
// Created by xabdomo on 4/16/25.
//

#include "logical_or.hpp"
#include "query/cpu/filter_applier.hpp"

#define OP_OR_DEBUG

tensor<char, CPU> * Ops::logical_or(
    FromResolver::ResolveResult *input_data,
    const hsql::Expr *eval,
    hsql::LimitDescription *limit) {
#ifdef OP_OR_DEBUG
    std::cout << "kExprOperator::OR" << std::endl;
#endif

    const auto left  = eval->expr;
    const auto right = eval->expr2;

    const auto left_result  = FilterApplier::apply(input_data, left,  limit);
    const auto right_result = FilterApplier::apply(input_data, right, limit);

    auto result = new tensor(*left_result || *right_result);

    delete left_result;
    delete right_result;

    return result;
}
