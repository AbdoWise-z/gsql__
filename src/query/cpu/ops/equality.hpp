//
// Created by xabdomo on 4/16/25.
//

#ifndef OP_EQUALITY_HPP
#define OP_EQUALITY_HPP

#include <hsql/sql/SelectStatement.h>

#include "query/cpu/from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace Ops {
    tensor<char, CPU>* equality(
        FromResolver::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit
    );
}


#endif //OP_EQUALITY_HPP
