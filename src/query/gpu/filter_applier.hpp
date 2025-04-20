//
// Created by xabdomo on 4/15/25.
//

#ifndef FILTER_APPLIER_HPP
#define FILTER_APPLIER_HPP

#include <hsql/sql/SelectStatement.h>

#include "from_resolver.hpp"
#include "tensor/tensor.hpp"


namespace FilterApplier::GPU {
     tensor<char, Device::CPU>* apply(
         FromResolver::GPU::ResolveResult *input_data,
         hsql::Expr* eval,                                       // the join / filter expression
         hsql::LimitDescription* limit,                          // max number of returned rows
         const std::vector<size_t>& tile_start = {},
         const std::vector<size_t>& tile_size = {}
    );
};



#endif //FILTER_APPLIER_HPP
