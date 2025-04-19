//
// Created by xabdomo on 4/16/25.
//

#ifndef LOGICAL_OR_HPP
#define LOGICAL_OR_HPP

#include "hsql/sql/SelectStatement.h"
#include "query/cpu/from_resolver.hpp"
#include "tensor/tensor.hpp"

namespace Ops {
    tensor<char, CPU>* logical_or(
        FromResolver::ResolveResult *input_data,
        const hsql::Expr *eval,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start = {},
        const std::vector<size_t>& tile_size = {}
    );
}

#endif //LOGICAL_OR_HPP
