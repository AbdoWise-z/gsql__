//
// Created by xabdomo on 4/16/25.
//

#ifndef SUM_HPP
#define SUM_HPP
#include "db/column.hpp"
#include "db/value_helper.hpp"


class table;

namespace Agg::CPU {
    tval sum(column* col);
    tval avg(column* col);
    tval count(const column* col);
    tval count(const table *t);
}



#endif //SUM_HPP
