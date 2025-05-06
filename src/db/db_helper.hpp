//
// Created by xabdomo on 4/13/25.
//

#ifndef DB_HELPER_HPP
#define DB_HELPER_HPP
#include <string>

#include "table.hpp"


namespace DBHelper {
    table* fromCSV(std::string path);

    table* fromCSV_Unchecked(std::string path);

    bool toCSV(table* t, const std::string& path);
}

#endif //DB_HELPER_HPP
