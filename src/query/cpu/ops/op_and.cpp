//
// Created by xabdomo on 4/16/25.
//

#include "op_and.hpp"

#include "query/cpu/filter_applier.hpp"

#define OP_AND_DEBUG

tensor<char, CPU> * Ops::logical_and(
    FromResolver::ResolveResult *input_data,
    hsql::Expr *eval,
    hsql::LimitDescription *limit) {

#ifdef OP_AND_DEBUG
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
