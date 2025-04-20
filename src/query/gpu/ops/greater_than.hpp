//
// Created by xabdomo on 4/16/25.
//

#ifndef GREATER_THAN_HPP
#define GREATER_THAN_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/gpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops::GPU {
    tensor<char, Device::GPU>* greater_than(
        FromResolver::GPU::ResolveResult *input_data,
        hsql::Expr *left,
        hsql::Expr *right,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start = {},
        const std::vector<size_t>& tile_size = {}
    );
}


#endif //GREATER_THAN_HPP
