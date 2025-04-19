//
// Created by xabdomo on 4/16/25.
//

#include "logical_and.hpp"

#include "query/cpu/filter_applier.hpp"

// #define OP_EQUALS_DEBUG

tensor<char, CPU> * Ops::logical_and(
    FromResolver::ResolveResult *input_data,
    const hsql::Expr *eval,
    hsql::LimitDescription *limit,
    const std::vector<size_t>& tile_start,
    const std::vector<size_t>& tile_size) {

#ifdef OP_EQUALS_DEBUG
    std::cout << "kExprOperator::AND" << std::endl;
#endif

    const auto left  = eval->expr;
    const auto right = eval->expr2;
    // set limit to always be nullptr because
    // count(left) AND count(right) <= count(this)
    // so I cannot limit either left or right since
    // I may undershoot the limit this way.
    const auto left_result  = FilterApplier::apply(input_data, left,  nullptr, tile_start, tile_size);
    const auto right_result = FilterApplier::apply(input_data, right, nullptr, tile_start, tile_size);

    auto result = new tensor(*left_result && *right_result);

    delete left_result;
    delete right_result;


    return result;
}
