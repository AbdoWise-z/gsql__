//
// Created by xabdomo on 4/13/25.
//

#ifndef TABLE_HPP
#define TABLE_HPP

#include <vector>
#include <string>

#include "column.hpp"

class table {
public:
    std::vector<std::string>  headers;
    std::vector<column*>      columns;

    table();
    table(std::vector<std::string>, std::vector<column*>);

    void setHeaders(std::vector<std::string> headers, const std::vector<DataType> &data_types);
    void addRecord(std::vector<tval> record);

    column& operator[] (size_t index) const;
    column& operator[] (const std::string& name);

    ~table();

    size_t size() const;


    static std::string details(table * table);
};



#endif //TABLE_HPP
