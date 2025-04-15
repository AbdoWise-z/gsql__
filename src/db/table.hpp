//
// Created by xabdomo on 4/13/25.
//

#ifndef TABLE_HPP
#define TABLE_HPP

#include <vector>
#include <string>

#include "column.hpp"
#include "value_helper.hpp"


class table {
public:
    std::vector<std::string> headers;
    std::vector<column>      columns;

    table();

    void setHeaders(std::vector<std::string> headers, const std::vector<DataType> &data_types);
    void addRecord(std::vector<tval> record);

    column& operator[] (size_t index);
    column& operator[] (const std::string& name);

    ~table();


    static std::string details(table * table);
};



#endif //TABLE_HPP
