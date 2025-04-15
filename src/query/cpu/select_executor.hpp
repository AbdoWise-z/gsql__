//
// Created by xabdomo on 4/15/25.
//

#ifndef SELECT_EXECUTOR_HPP
#define SELECT_EXECUTOR_HPP

#include <hsql/sql/SQLStatement.h>


class table;

namespace SelectExecutor {
     table* Execute(hsql::SQLStatement* statement);
};



#endif //SELECT_EXECUTOR_HPP
