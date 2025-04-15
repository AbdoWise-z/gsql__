//
// Created by xabdomo on 4/15/25.
//

#ifndef FILTER_APPLIER_HPP
#define FILTER_APPLIER_HPP

#include <hsql/sql/SelectStatement.h>

#include "store.hpp"
#include "db/table.hpp"
#include "tensor/tensor.hpp"


namespace FilterApplier {
     tensor<char, CPU>* apply(
        std::unordered_map<std::string, table*>& tables,        // what are the tables involved ?
        hsql::Expr* eval,                                       // the join / filter expression
        hsql::LimitDescription* limit,                          // max number of returned rows
        const std::vector<std::string>& ordered_tables          // since unordered_map doesn't retain ordering.
    );
};



#endif //FILTER_APPLIER_HPP
