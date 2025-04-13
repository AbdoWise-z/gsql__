//
// Created by xabdomo on 4/13/25.
//

#include "table.hpp"

#include <algorithm>
#include <c++/12/stdexcept>

#include "column.hpp"

table::table() {
    headers = {};
    columns = {};
}

void table::setHeaders(std::vector<std::string> headers, const std::vector<DataType> &data_types) {
    if (headers.size() != data_types.size()) {
        throw std::invalid_argument("header size != data_types size");
    }

    this->headers = headers;
    for (int i = 0;i < headers.size();i++) {
        column c;
        c.type = data_types[i];
        columns.push_back(c);
    }
}

void table::addRecord(std::vector<tval> record) {
    if (record.size() != columns.size()) {
        throw std::invalid_argument("record size does not match column size");
    }

    for (int i = 0;i < record.size();i++) {
        columns[i].data.push_back(record[i]);
    }
}

column & table::operator[](size_t index) {
    return columns[index];
}

column & table::operator[](const std::string &name) {
    return columns[std::distance(headers.begin(), std::ranges::find(headers, name))];
}
