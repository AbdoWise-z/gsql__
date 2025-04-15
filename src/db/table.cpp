//
// Created by xabdomo on 4/13/25.
//

#include "table.hpp"

#include <algorithm>
#include <iomanip>
#include <c++/12/stdexcept>

#include "column.hpp"
#include "utils/konsol.hpp"

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

table::~table() = default;

std::string table::details(table * table) {
    std::stringstream ss;

    ss << color(std::to_string(table->headers.size()), RED_FG) << " columns";
    if (!table->columns.empty()) {
        ss << ", " << color(std::to_string(table->columns[0].data.size()), RED_FG) << " rows";
    }
    ss << std::endl;

    for (int i = 0; i < table->headers.size(); i++) {
        ss << "| -- "
            << std::setw(8) << std::left
            << "(" + typeToString(table->columns[i].type) + ")"
            << color(table->headers[i], GREEN_FG) << std::endl;
    }

    return ss.str();
}
