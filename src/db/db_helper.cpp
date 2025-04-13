//
// Created by xabdomo on 4/13/25.
//

#include "db_helper.hpp"

#include <csv.hpp>
#include <hsql/sql/ColumnType.h>


static std::vector<DataType> inferTypes(const csv::CSVRow& row) {
    std::vector<DataType> types;
    for (auto field: row) {
        switch (field.type()) {
            case csv::DataType::CSV_STRING:
                types.push_back(STRING);
                break;
            case csv::DataType::CSV_INT8:
            case csv::DataType::CSV_INT16:
            case csv::DataType::CSV_INT32:
            case csv::DataType::CSV_INT64:
                types.push_back(INTEGER);
                break;
            case csv::DataType::CSV_DOUBLE:
                types.push_back(FLOAT);

            default:
                throw std::runtime_error("Not implemented");
        }
    }

    return types;
}

table * fromCSV(std::string path) {
    csv::CSVReader csv(path);
    auto headers = csv.get_col_names();
    auto table   = new ::table();

    bool first_row = true;
    std::vector<DataType> types;
    for (const auto& row : csv) {
        std::vector<tval> values;
        if (first_row) {
            types = inferTypes(row);
            first_row = false;

            table->setHeaders(headers, types);
        }

        for (auto field: row) {
            tval val;
            switch (field.type()) {
                case csv::DataType::CSV_STRING:
                    val.s = field.get<std::string>().c_str();
                    break;
                case csv::DataType::CSV_INT8:
                case csv::DataType::CSV_INT16:
                case csv::DataType::CSV_INT32:
                case csv::DataType::CSV_INT64:
                    val.i = field.get<int64_t>();
                    break;
                case csv::DataType::CSV_DOUBLE:
                    val.d = field.get<double>();
                    break;
                default:
                    throw std::runtime_error("Not implemented");
            }
            values.push_back(val);
        }

        table->addRecord(values);
    }

    return table;
}
