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
        std::unordered_map<std::string, table*>& tables,
        hsql::Expr* eval,
        hsql::LimitDescription* limit
    );
};



#endif //FILTER_APPLIER_HPP
