//
// Created by xabdomo on 4/15/25.
//

#ifndef FILTER_APPLIER_HPP
#define FILTER_APPLIER_HPP

#include <hsql/sql/SelectStatement.h>

#include "from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace FilterApplier {
     tensor<char, CPU>* apply(
         FromResolver::ResolveResult *input_data,
         hsql::Expr* eval,                                       // the join / filter expression
         hsql::LimitDescription* limit                           // max number of returned rows
    );
};



#endif //FILTER_APPLIER_HPP
