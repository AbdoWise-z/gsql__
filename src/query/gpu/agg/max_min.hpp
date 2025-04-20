//
// Created by xabdomo on 4/16/25.
//

#ifndef MAX_HPP
#define MAX_HPP
#include "db/column.hpp"
#include "db/value_helper.hpp"


namespace Agg::GPU {
    tval max(column* col);
    tval min(column* col);
}



#endif //MAX_HPP
