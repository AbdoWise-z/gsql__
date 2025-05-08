//
// Created by xabdomo on 4/16/25.
//

#ifndef OP_NULL_EQUALITY_HPP
#define OP_NULL_EQUALITY_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/cpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops::CPU {
    tensor<char, Device::CPU>* null_equality(
        FromResolver::CPU::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start = {},
        const std::vector<size_t>& tile_size = {}
    );
}


#endif //OP_NULL_EQUALITY_HPP
