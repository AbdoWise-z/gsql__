//
// Created by xabdomo on 4/16/25.
//

#ifndef GREATER_THAN_HPP
#define GREATER_THAN_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/cpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops::CPU {
    tensor<char, Device::CPU>* greater_than(
        FromResolver::CPU::ResolveResult *input_data,
        hsql::Expr *left,
        hsql::Expr *right,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start = {},
        const std::vector<size_t>& tile_size = {}
    );
}


#endif //GREATER_THAN_HPP
