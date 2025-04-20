//
// Created by xabdomo on 4/16/25.
//

#ifndef OP_EQUALITY_HPP
#define OP_EQUALITY_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/gpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops::GPU {
    tensor<char, Device::GPU>* equality(
        FromResolver::GPU::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start = {},
        const std::vector<size_t>& tile_size = {}
    );
}


#endif //OP_EQUALITY_HPP
