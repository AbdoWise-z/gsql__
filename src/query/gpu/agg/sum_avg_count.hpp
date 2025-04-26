//
// Created by xabdomo on 4/16/25.
//

#ifndef SUM_HPP
#define SUM_HPP
#include "db/column.hpp"
#include "db/table.hpp"
#include "db/value_helper.hpp"


namespace Agg::GPU {
    tval sum(column* col);
    tval avg(column* col);
    tval count(const column* col);
    tval count(const table* t);
}



#endif //SUM_HPP
