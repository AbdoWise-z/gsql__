
//
// Created by xabdomo on 4/15/25.
//

#ifndef GPU_EXECUTOR_HPP
#define GPU_EXECUTOR_HPP

#include <hsql/SQLParserResult.h>

#include "store.hpp"
#include "db/table.hpp"


namespace GPUExecutor {
    std::vector<table*> executeQuery(const hsql::SQLParserResult&, TableMap& tables);
};



#endif //GPU_EXECUTOR_HPP
