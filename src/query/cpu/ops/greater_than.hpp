//
// Created by xabdomo on 4/16/25.
//

#ifndef GREATER_THAN_HPP
#define GREATER_THAN_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/cpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops {
    tensor<char, CPU>* greater_than(
        FromResolver::ResolveResult *input_data,
        hsql::Expr *left,
        hsql::Expr *right,
        hsql::LimitDescription *limit
    );
}


#endif //GREATER_THAN_HPP
