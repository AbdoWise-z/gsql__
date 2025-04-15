//
// Created by xabdomo on 4/15/25.
//

#ifndef CPU_EXECUTOR_HPP
#define CPU_EXECUTOR_HPP
#include <hsql/SQLParserResult.h>

#include "db/table.hpp"


namespace CPUExecutor {
    std::vector<table*> executeQuery(const hsql::SQLParserResult&);
};



#endif //CPU_EXECUTOR_HPP
