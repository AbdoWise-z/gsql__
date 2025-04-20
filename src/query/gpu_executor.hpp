
//
// Created by xabdomo on 4/15/25.
//

#ifndef GPU_EXECUTOR_HPP
#define GPU_EXECUTOR_HPP

#include <hsql/SQLParserResult.h>

#include "db/table.hpp"


namespace GPUExecutor {
    std::vector<table*> executeQuery(const hsql::SQLParserResult&);
};



#endif //CPU_EXECUTOR_HPP
